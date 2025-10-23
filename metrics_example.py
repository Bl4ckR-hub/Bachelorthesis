import torch
import Modells
import Criterion
import piq
import CT_library
import Reconstructor
import Metrics

#######You can Modify her
train_dir = '/user/viktor.tevosyan/u17320/project/LoDoPaB'
valid_dir = '/user/viktor.tevosyan/u17320/project/LoDoPaB'
best_model_checkpoint = '/user/viktor.tevosyan/u17320/project/Codebase/windowII+edge+perceptual+subset/best15.pth'
latest_model_checkpoint = '/user/viktor.tevosyan/u17320/project/Codebase/windowII+edge+perceptual+subset/last5.pth'
training_results_dir = '/user/viktor.tevosyan/u17320/project/Codebase/windowII+edge+perceptual+subset/'
eps=1e-9

device = 0 if torch.cuda.is_available() else torch.device('cpu')
#######

def main(checkpoint_path, eval_results_path):
    # Load model
    window = Modells.LearnableWindowII()
    filter_model = Reconstructor.Ramp_Filter()
    filtering_module = Reconstructor.Filtering_Module(filter_model=filter_model, window_model=window)
    vanilla_backproj = Reconstructor.Vanilla_Backproj()
    
    fbp = Reconstructor.LearnableFBP(filtering_module=filtering_module, backprojection_module=vanilla_backproj, post_processing_module=torch.nn.Identity()).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    fbp.load_state_dict(checkpoint['model_state_dict'])

    # Load datasets
    train_dataset = CT_library.LoDoPaB_Dataset(sino_dir=train_dir, gt_images_dir=train_dir, suffix='train', amount_images=1000)
    valid_dataset = CT_library.LoDoPaB_Dataset(sino_dir=valid_dir, gt_images_dir=valid_dir, suffix='valid', amount_images=1000)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False)
    
    metrics = {'mean_psnr_train': [],
               'mean_ssim_train': [],
               'mean_psnr_val': [],
               'mean_ssim_val': [],
               'best_val_psnr': (0,0),
               'best_val_ssim': (0,0),
               'best_psnr_train': (0,0),
               'best_ssim_train': (0,0),
               'worst_psnr_train': (float('inf'),0),
               'worst_ssim_train': (float('inf'),0),
               'worst_val_psnr': (float('inf'),0),
               'worst_val_ssim': (float('inf'),0)
               }

    # Evaluate first 1000 images
    def evaluate(dataset, split_name):
        for i, data in enumerate(dataset):
            print(f'Step: {i} of {len(dataset)}')
            input_img = data[0].to(device)
            target_img = data[1].to(device)
            with torch.no_grad():
                output = fbp(input_img)
                psnr = Metrics.psnr(output, target_img)
                ssim = Metrics.ssim_metric(torch.clamp(output, 0, 1), target_img)
                

                if split_name == "Train":
                    metrics['mean_psnr_train'].append(psnr.item())
                    metrics['mean_ssim_train'].append(ssim.item())

                    if psnr.item() > metrics['best_psnr_train'][0]:
                        metrics['best_psnr_train'] = (psnr.item(), i)
                    if ssim.item() > metrics['best_ssim_train'][0]:
                        metrics['best_ssim_train'] = (ssim.item(), i)

                    if psnr.item() < metrics['worst_psnr_train'][0]:
                        metrics['worst_psnr_train'] = (psnr.item(), i)
                    if ssim.item() < metrics['worst_ssim_train'][0]:
                        metrics['worst_ssim_train'] = (ssim.item(), i)

                else:
                    metrics['mean_psnr_val'].append(psnr.item())
                    metrics['mean_ssim_val'].append(ssim.item())

                    if psnr.item() > metrics['best_val_psnr'][0]:
                        metrics['best_val_psnr'] = (psnr.item(), i)
                    if ssim.item() > metrics['best_val_ssim'][0]:
                        metrics['best_val_ssim'] = (ssim.item(), i)
                

                    if psnr.item() < metrics['worst_val_psnr'][0]:
                        metrics['worst_val_psnr'] = (psnr.item(), i)
                    if ssim.item() < metrics['worst_val_ssim'][0]:
                        metrics['worst_val_ssim'] = (ssim.item(), i)



    evaluate(train_loader, "Train")
    evaluate(valid_loader, "Validation")
    metrics['mean_psnr_train'] = sum(metrics['mean_psnr_train']) / len(metrics['mean_psnr_train'])
    metrics['mean_ssim_train'] = sum(metrics['mean_ssim_train']) / len(metrics['mean_ssim_train'])
    metrics['mean_psnr_val'] = sum(metrics['mean_psnr_val']) / len(metrics['mean_psnr_val'])
    metrics['mean_ssim_val'] = sum(metrics['mean_ssim_val']) / len(metrics['mean_ssim_val'])

    torch.save(metrics, eval_results_path)
    print("Evaluation metrics saved.")  

if __name__ == "__main__":
    main(best_model_checkpoint, training_results_dir + 'evaluation_best15.pth')

