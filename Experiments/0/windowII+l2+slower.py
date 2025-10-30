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
best_model_checkpoint = '/user/viktor.tevosyan/u17320/project/Codebase/windowII+l2+slower/best.pth'
latest_model_checkpoint = '/user/viktor.tevosyan/u17320/project/Codebase/windowII+l2+slower/last.pth'
training_results_dir = '/user/viktor.tevosyan/u17320/project/Codebase/windowII+l2+slower/'

device = 0 if torch.cuda.is_available() else torch.device('cpu')
#######



def overfitter_preprocessing(X, Y, device, model, optimizer, criterion, epochs, eps=1e-4, max_counter=5, min_lr=1e-6):
    model = model.to(device)
    X = X.to(device)
    Y = Y.to(device)

    losses = []
    best_loss = float('inf')
    counter = 0
    stop = False


    for epoch in range(epochs):
        if stop is True:
            break
        print(f"Entering Epoch {epoch}")

        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred,Y)
        loss.backward()

        if abs(loss.item() - best_loss) <= eps or loss.item() > best_loss:
            counter += 1
        else:
            counter = 0
            save_params(model.filtering_module.window_model, training_results_dir, 0)
            best_loss = loss.item()

        if counter >= max_counter:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10# Set the new learning rate
                if param_group['lr'] <= min_lr:
                    stop = True
                print(f'lr = {param_group['lr']}')
            model.filtering_module.window_model.load_state_dict(torch.load(training_results_dir + 'params0.pth'))
            counter = 0

        optimizer.step()


        losses.append(loss.item())

        print(f"Finished Epoch {epoch} with loss: {loss.item()}")
        
    return losses

def evaluator(X,Y,model, device):
    X = X.to(device)
    Y = Y.to(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        pred = model(X)

    psnr = Metrics.psnr(pred, Y)
    l1 = Metrics.l1_loss(pred, Y)
    mse = Metrics.mse_loss(pred, Y)
    ssim = Metrics.ssim_metric(torch.clamp(pred, 0, 1), torch.clamp(Y, 0, 1))

    return psnr, l1, mse , ssim

def save_params(model, dire, i=1):
    torch.save(model.state_dict(), dire + f'params{i}.pth')


def main():
    l2 = torch.nn.MSELoss(reduction="sum")


    dataset = CT_library.LoDoPaB_Dataset(sino_dir=train_dir, gt_images_dir=train_dir, suffix='train', amount_images=1000)

    window = Modells.LearnableWindowII()
    filter_model = Reconstructor.Ramp_Filter()
    filtering_module = Reconstructor.Filtering_Module(filter_model=filter_model, window_model=window)
    vanilla_backproj = Reconstructor.Vanilla_Backproj()
    fbp = Reconstructor.LearnableFBP(filtering_module=filtering_module, backprojection_module=vanilla_backproj, post_processing_module=torch.nn.Identity())
    
    optimizer = torch.optim.Adam(window.parameters(), lr=1e-3)
    X,Y = dataset[50]
    X = X.unsqueeze(0)
    Y = Y.unsqueeze(0)

    losses = overfitter_preprocessing(X, Y, device, fbp, optimizer, l2, 3000)

    torch.save(losses, training_results_dir + 'losses.pth')

    

main()