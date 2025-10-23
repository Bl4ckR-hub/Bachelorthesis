import torch
import os

import CT_library
import Metrics
import torch.distributed as dist

MAX = float("inf")





class Trainer:
    def __init__(self, model, train_loader, valid_loader, optim, criterion, criterion_sinogram=None, best_model_checkpoint=None,
                 latest_model_checkpoint=None, lr_scheduler=None, rank=torch.device('cpu'), one_cycle_lr=False, training_results_dir=None):
        print("Initializing Variables")
        # Init vars
        self.rank = rank
        self.optimizer = optim
        self.best_model_checkpoint = best_model_checkpoint
        self.latest_model_checkpoint = latest_model_checkpoint
        self.training_results_dir = training_results_dir
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr_scheduler = lr_scheduler
        self.one_cycle_lr = one_cycle_lr
        self.criterion_sinogram = criterion_sinogram

        self.model = model
        self.criterion = criterion

        self.radon = CT_library.RadonTransform(rank).radon
        print("Initing result arrays")
        self.train_losses = []
        self.val_losses = []
        self.metrics = []

        self.val_best = MAX
        self.stop_training = False
        self.early_stopper_counter = 0
        print("Trainer init completed")


    def _run_train_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss1 = self.criterion(output, targets)
        loss2 = torch.tensor(0.0, device=self.rank)
        if self.criterion_sinogram is not None:
            targets_sino = self.radon(targets).to(self.rank)
            output_sino = self.radon(output).to(self.rank)
            loss2 = self.criterion_sinogram(targets_sino, output_sino)


        loss = loss1 + loss2

        loss.backward()
        self.optimizer.step()

        if self.one_cycle_lr:
            self.lr_scheduler.step()

        return loss.item()

    def _run_valid_batch(self, source, targets):
        output = self.model(source)
        loss1 = self.criterion(output, targets)
        loss2 = torch.tensor(0.0, device=self.rank)
        if self.criterion_sinogram is not None:
            targets_sino = self.radon(targets).to(self.rank)
            output_sino = self.radon(output).to(self.rank)
            loss2 = self.criterion_sinogram(targets_sino, output_sino)

        loss = loss1 + loss2
        return loss.item()

    def _run_train_epoch(self, epoch):

        b_sz = len(next(iter(self.train_loader))[0])
        print(f"[GPU{self.rank}] Training | Epoch: {epoch + 1} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}")
        running_loss = 0
        self.model.train()
        self.train_loader.sampler.set_epoch(epoch)
        for num, (source, targets) in enumerate(self.train_loader):
            source = source.to(self.rank)
            targets = targets.to(self.rank)
            running_loss += self._run_train_batch(source, targets)

        total_loss = torch.tensor(running_loss).to(self.rank)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)  # Sum losses from all GPUs
        epoch_loss = (total_loss / (len(self.train_loader) * dist.get_world_size())).item()

        if self.rank == 0:  # Only store on master process
            self.train_losses.append(epoch_loss)

        print(f"[GPU{self.rank}] Training | Epoch: {epoch + 1} finished with loss: {epoch_loss}")

    def _run_valid_epoch(self, epoch):
                
        b_sz = len(next(iter(self.valid_loader))[0])
        print(f"[GPU{self.rank}] Validation | Epoch: {epoch + 1} | Batchsize: {b_sz} | Steps: {len(self.valid_loader)}")
        running_loss = 0

        self.model.eval()
        with torch.no_grad():
            for num, (source, targets) in enumerate(self.valid_loader):
                source = source.to(self.rank)
                targets = targets.to(self.rank)
                running_loss += self._run_valid_batch(source, targets)

        total_loss = torch.tensor(running_loss).to(self.rank)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)  # Sum losses from all GPUs
        epoch_loss = (total_loss / (len(self.valid_loader) * dist.get_world_size())).item()

        if not self.one_cycle_lr and self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch_loss)

        if self.rank == 0:
            self.val_losses.append(epoch_loss)
            if epoch_loss < self.val_best:
                self._save_checkpoint(self.best_model_checkpoint)
                self.val_best = epoch_loss

        print(f"[GPU{self.rank}] Validation | Epoch: {epoch + 1} finished with loss: {epoch_loss}")

    def _save_checkpoint(self, checkpoint_dir):
        checkpoint = {
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_best': self.val_best
        }
        if os.path.exists(checkpoint_dir):
            os.remove(checkpoint_dir)
        torch.save(checkpoint, checkpoint_dir)

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            
            ###
            if self.rank == 0:
                if epoch > 0 and epoch % 5 == 0:
                    dir_to_save = self.training_results_dir + f'checkpoint_epoch_{epoch}.pth'
                    self._save_checkpoint(dir_to_save)
            ###



            if self.stop_training:
                break

            print(f"Entering Epoch {epoch + 1}")
            self._run_train_epoch(epoch)
            self._run_valid_epoch(epoch)
            print(f"End of Epoch {epoch + 1}")
        return {'train_losses': self.train_losses, 'valid_losses': self.val_losses}

    def evaluate(self):
        print(f"Entering Evaluation")
        self.model.eval()

        ls_psnr, ls_l1, ls_mse, ls_ssim = [], [], [], []

        with torch.no_grad():
            for num, (source, targets) in enumerate(self.valid_loader):
                source = source.to(self.rank)
                outputs = self.model(source)
                targets = targets.to(self.rank)
                psnr, l1, mse, ssim = self._evaluator(outputs, targets)
                ls_psnr.append(psnr)
                ls_l1.append(l1)
                ls_mse.append(mse)
                ls_ssim.append(ssim)

        ls_psnr = torch.stack(ls_psnr)
        ls_l1 = torch.stack(ls_l1)
        ls_mse = torch.stack(ls_mse)
        ls_ssim = torch.stack(ls_ssim)

        dist.all_reduce(ls_psnr, op=dist.ReduceOp.SUM)
        dist.all_reduce(ls_l1, op=dist.ReduceOp.SUM)
        dist.all_reduce(ls_mse, op=dist.ReduceOp.SUM)
        dist.all_reduce(ls_ssim, op=dist.ReduceOp.SUM)

        world_size = dist.get_world_size()
        ls_psnr /= world_size
        ls_l1 /= world_size
        ls_mse /= world_size
        ls_ssim /= world_size

        self.metrics = {'mean_psnr': ls_psnr.mean(), 'mean_l1': ls_l1.mean(), 'mean_mse': ls_mse.mean(),
                        'mean_ssim': ls_ssim.mean(),
                        'best_psnr': (ls_psnr.max(), ls_psnr.argmax()), 'best_l1': (ls_l1.min(), ls_l1.argmin()),
                        'best_mse': (ls_mse.min(), ls_mse.argmin()), 'best_ssim': (ls_ssim.max(), ls_ssim.argmax()),
                        'worst_psnr': (ls_psnr.min(), ls_psnr.argmin()), 'worst_l1': (ls_l1.max(), ls_l1.argmax()),
                        'worst_mse': (ls_mse.max(), ls_mse.argmax()), 'worst_ssim': (ls_ssim.min(), ls_ssim.argmin())}

    def _evaluator(self, pred, target):

        psnr = Metrics.psnr(pred, target)
        l1 = Metrics.l1_loss(pred, target)
        mse = Metrics.mse_loss(pred, target)
        ssim = Metrics.ssim_metric(torch.clamp(pred, 0, 1), torch.clamp(target, 0, 1))

        return psnr, l1, mse, ssim
    


