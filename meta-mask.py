import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import lightly
from loss import BarlowTwinsLoss
import sys
import copy
import argparse

from utils import knn_predict, BenchmarkModule
from models.mask_generator import FeatureMask
from models.simsiam import MaskSimSiam
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Meta Mask Training for SimCLR on Cifar10')
parser.add_argument('--projection-out-dim', default=2048, type=int,
                    help='dimension of projection head in Barlowtwins')
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--max-epochs', default=800, type=int)
parser.add_argument('--knn_k', default=5, type=int)
parser.add_argument('--knn_t', default=1.0, type=float)
parser.add_argument('--classes', default=10, type=int)
parser.add_argument('--batch-size', default=512, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--cl-projection-out-dim', default=256, type=int, metavar='N',
                    help='dimension of projection head in SimCLR')
parser.add_argument('--disable-meta', action="store_true",
                    help='whether using meta learning when training')
parser.add_argument('--disable-sigmoid',
                    action="store_true",
                    help='whether using meta learning when training')
parser.add_argument('--no-second-order', action="store_true")
parser.add_argument('--temperature', default=0.07, type=float,
                    help='temperature of simclr loss')
parser.add_argument('--weight-cl', default=10, type=float,
                    help='weight of simclr loss')
parser.add_argument('--ckpt-path', type=str, default=None)
parser.add_argument('--eval-only', action="store_true")


def main():
    args = parser.parse_args()
    pl.seed_everything(args.seed)

    # use a GPU if available
    gpus = 1 if torch.cuda.is_available() else 0
    device = 'cuda' if gpus else 'cpu'

    input_img_size=32
    # Use SimCLR augmentations, additionally, disable blur
    collate_fn = lightly.data.SimCLRCollateFunction(
        input_size=input_img_size,
        gaussian_blur=0.,
    )

    # No additional augmentations for the test set
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])

    dataset_train_ssl = lightly.data.LightlyDataset.from_torch_dataset(
        torchvision.datasets.CIFAR10(
            root='data',
            train=True,
            download=True))
    dataset_train_kNN = lightly.data.LightlyDataset.from_torch_dataset(torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        transform=test_transforms,
        download=True))
    dataset_test = lightly.data.LightlyDataset.from_torch_dataset(torchvision.datasets.CIFAR10(
        root='data',
        train=False,
        transform=test_transforms,
        download=True))

    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=args.num_workers
    )
    dataloader_train_kNN = torch.utils.data.DataLoader(
        dataset_train_kNN,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )
    model = MetaMaskBartonTwins(dataloader_train_kNN, gpus=gpus,
                                classes=args.classes, knn_k=args.knn_k, knn_t=args.knn_t,
                                projection_out_dim=args.projection_out_dim,
                                cl_projection_out_dim=args.cl_projection_out_dim, device=device,
                                enable_meta=not args.disable_meta,
                                enable_sigmoid=not args.disable_sigmoid, weight_cl=args.weight_cl,
                                max_epochs=args.max_epochs,
                                second_order=not args.no_second_order,
                                eval_only=args.eval_only
                                )

    trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=gpus,
                         progress_bar_refresh_rate=1)

    if args.eval_only:
        trainer.validate(
            model,
            dataloaders=dataloader_test,
            ckpt_path=args.ckpt_path
        )
    else:
        trainer.fit(
            model,
            train_dataloaders=dataloader_train_ssl,
            val_dataloaders=dataloader_test
        )

    print(f'Highest test accuracy: {model.max_accuracy:.4f}')

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

class MetaMaskBartonTwins(BenchmarkModule):
    def __init__(self, dataloader_kNN, gpus, classes, knn_k, knn_t,
                 projection_out_dim, cl_projection_out_dim, device,
                 enable_meta, enable_sigmoid, weight_cl, max_epochs, second_order, eval_only):
        super().__init__(dataloader_kNN, gpus, classes, knn_k, knn_t)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        self.resnet_simsiam = \
            MaskSimSiam(self.backbone, num_ftrs=512, num_mlp_layers=3, out_dim=projection_out_dim,
                           aux_num_mlp_layers=2, aux_out_dim=cl_projection_out_dim)
        self.resnet_simsiam_ = copy.deepcopy(self.resnet_simsiam)

        self.criterion = BarlowTwinsLoss(device=device)
        if enable_meta:
            self.auto_mask = FeatureMask(512, enable_sigmoid)
            self.automatic_optimization = False
        self.temperature = 0.07

        self.enable_meta = enable_meta
        self.weight_cl = weight_cl
        self.max_epochs = max_epochs
        self.second_order = second_order
        self.eval_only = eval_only

    # def forward(self, x):
    #     self.resnet_simsiam(x)

    def unrolled_backward(self, x0, x1, model_optim, mask_optim, eta):
        """
        Compute un-rolled loss and backward its gradients
        """
        #  compute unrolled multi-task network theta_1^+ (virtual step)
        masks = self.auto_mask()
        _, _, x0_cl, x1_cl = self.resnet_simsiam(x0, x1, masks)
        # our simsiam model returns both (features + projection head)
        z_a_cl, _ = x0_cl
        z_b_cl, _ = x1_cl
        loss = self.contrastive_loss(z_a_cl, z_b_cl)

        model_optim.zero_grad()
        mask_optim.zero_grad()
        self.manual_backward(loss)
        gradients = copy.deepcopy([v.grad.data if v.grad is not None else None for v in self.resnet_simsiam.parameters()])

        model_optim.zero_grad()
        mask_optim.zero_grad()
        # do virtual step: theta_1^+ = theta_1 - alpha * (primary loss + auxiliary loss)
        with torch.no_grad():
            for weight, weight_, d_p in zip(self.resnet_simsiam.parameters(),
                                                   self.resnet_simsiam_.parameters(),
                                                   gradients):
                # print(d_p)
                g = model_optim.param_groups[0]

                if d_p is None:
                    weight_.copy_(weight)
                    continue

                if g['weight_decay'] != 0:
                    d_p = d_p.add(weight, alpha=g['weight_decay'])

                if g['momentum'] != 0:
                    state = model_optim.state[weight]
                    if 'momentum_buffer' not in state:
                        buf = torch.clone(d_p).detach()
                    else:
                        buf = copy.deepcopy(state['momentum_buffer'])
                        buf.mul_(g['momentum']).add_(d_p, alpha=1)
                weight_.copy_(weight - buf * g['lr'])
                weight_.grad = None

        masks = self.auto_mask()
        _, _, x0_cl, x1_cl = self.resnet_simsiam_(x0, x1, masks)
        z_a_cl, _ = x0_cl
        z_b_cl, _ = x1_cl
        loss = self.contrastive_loss(z_a_cl, z_b_cl)

        mask_optim.zero_grad()
        self.manual_backward(loss)

        dalpha = [v.grad for v in self.auto_mask.parameters()]
        if self.second_order:
            vector = [v.grad.data if v.grad is not None else None for v in self.resnet_simsiam_.parameters()]

            implicit_grads = self._hessian_vector_product(vector, x0, x1)

            for g, ig in zip(dalpha, implicit_grads):
                g.data.sub_(ig.data, alpha=eta)

        for v, g in zip(self.auto_mask.parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _hessian_vector_product(self, gradients, x0, x1, r=1e-2):
        with torch.no_grad():
            for weight, weight_ in zip(self.resnet_simsiam.parameters(), self.resnet_simsiam_.parameters()):
                weight_.copy_(weight)
                weight_.grad = None

        valid_grad = []
        for grad in gradients:
            if grad is not None:
                valid_grad.append(grad)
        R = r / _concat(valid_grad).norm()
        for p, v in zip(self.resnet_simsiam_.parameters(), gradients):
            if v is not None:
                p.data.add_(v, alpha=R)

        masks = self.auto_mask()
        _, _, x0_cl, x1_cl = self.resnet_simsiam_(x0, x1, masks)
        z_a_cl, _ = x0_cl
        z_b_cl, _ = x1_cl
        loss = self.contrastive_loss(z_a_cl, z_b_cl)
        grads_p = torch.autograd.grad(loss, self.auto_mask.parameters())

        for p, v in zip(self.resnet_simsiam_.parameters(), gradients):
            if v is not None:
                p.data.sub_(v, alpha=2 * R)

        masks = self.auto_mask()
        _, _, x0_cl, x1_cl = self.resnet_simsiam_(x0, x1, masks)
        z_a_cl, _ = x0_cl
        z_b_cl, _ = x1_cl
        loss = self.contrastive_loss(z_a_cl, z_b_cl)

        grads_n = torch.autograd.grad(loss, self.auto_mask.parameters())

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    def contrastive_loss(self, x0, x1, norm=True):
        # https://github.com/google-research/simclr/blob/master/objective.py
        bsize = x0.shape[0]
        target = torch.arange(bsize).cuda()
        eye_mask = torch.eye(bsize).cuda() * 1e9
        if norm:
            x0 = F.normalize(x0, p=2, dim=1)
            x1 = F.normalize(x1, p=2, dim=1)
        logits00 = x0 @ x0.t() / self.temperature - eye_mask
        logits11 = x1 @ x1.t() / self.temperature - eye_mask
        logits01 = x0 @ x1.t() / self.temperature
        logits10 = x1 @ x0.t() / self.temperature
        return (
                       F.cross_entropy(torch.cat([logits01, logits00], dim=1), target)
                       + F.cross_entropy(torch.cat([logits10, logits11], dim=1), target)
               ) / 2

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch

        masks = torch.ones((512)).to(x0.device)
        if self.enable_meta:
            masks = self.auto_mask()

        x0_bl, x1_bl, x0_cl, x1_cl = self.resnet_simsiam(x0, x1, masks)
        # our simsiam model returns both (features + projection head)
        z_a_bl, _ = x0_bl
        z_b_bl, _ = x1_bl
        z_a_cl, _ = x0_cl
        z_b_cl, _ = x1_cl

        loss_bl = self.criterion(z_a_bl, z_b_bl)
        self.log('train_loss_bl', loss_bl)

        loss_cl = self.contrastive_loss(z_a_cl, z_b_cl)
        self.log('train_loss_cl', loss_cl)

        loss = loss_bl + self.weight_cl * loss_cl

        if self.enable_meta:
            opt_a, opt_b = self.optimizers(use_pl_optimizer=True)
            sche_a, _ = self.lr_schedulers()
            self.per_optimizer_step(opt_a, None, loss)
            self.unrolled_backward(x0, x1, opt_a, opt_b, sche_a.get_last_lr()[0])
            self.per_optimizer_step(opt_b)

        return loss

    def per_optimizer_step(self,
                        optimizer_a=None,
                        optimizer_b=None,
                        loss=None):

        # update params
        if loss is not None:
            optimizer_a.zero_grad()
            if optimizer_b is not None:
                optimizer_b.zero_grad()
            self.manual_backward(loss)

        optimizer_a.step()
        optimizer_a.zero_grad()
        if optimizer_b is not None:
            optimizer_b.step()
            optimizer_b.zero_grad()

    def configure_optimizers(self):

        optim = torch.optim.SGD(self.resnet_simsiam.parameters(), lr=1e-3,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)

        if self.enable_meta:

            mask_optim = torch.optim.SGD(self.auto_mask.parameters(), lr=1e-3,
                                    momentum=0.9, weight_decay=5e-4)
            mask_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(mask_optim, self.max_epochs)
            return [optim, mask_optim], [scheduler, mask_scheduler]
        else:
            return [optim], [scheduler]

    def training_epoch_end(self, outputs):
        # update feature bank at the end of each training epoch
        self.backbone.eval()
        self.feature_bank = []
        self.targets_bank = []
        with torch.no_grad():
            for data in self.dataloader_kNN:
                img, target, _ = data
                if self.gpus > 0:
                    img = img.cuda()
                    target = target.cuda()
                feature = self.backbone(img).squeeze()
                if self.enable_meta:
                    masks = self.auto_mask().detach()
                    feature = feature * masks[None, :]
                feature = F.normalize(feature, dim=1)
                self.feature_bank.append(feature)
                self.targets_bank.append(target)
        self.feature_bank = torch.cat(self.feature_bank, dim=0).t().contiguous()
        self.targets_bank = torch.cat(self.targets_bank, dim=0).t().contiguous()
        self.backbone.train()

        if not self.eval_only:
            if self.enable_meta:
                sche_a, sche_b = self.lr_schedulers()
                sche_a.step()
                sche_b.step()

    def validation_step(self, batch, batch_idx):
        if self.eval_only:
            if not hasattr(self, 'feature_bank') or not hasattr(self, 'targets_bank'):
                self.training_epoch_end(None)
                self.backbone.eval()
        # we can only do kNN predictions once we have a feature bank
        if hasattr(self, 'feature_bank') and hasattr(self, 'targets_bank'):
            images, targets, _ = batch
            feature = self.backbone(images).squeeze()
            if self.enable_meta:
                masks = self.auto_mask().detach()
                feature = feature * masks[None, :]
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(feature, self.feature_bank, self.targets_bank, self.classes, self.knn_k,
                                      self.knn_t)
            num = images.size(0)
            top1 = (pred_labels[:, 0] == targets).float().sum().item()
            return (num, top1)

if __name__ == '__main__':
    main()

