import time
import tqdm
import wandb
import torch

def train_epoch(dataloader, 
                model, 
                optimizer,  
                obj_id,
                device
                ):
    
    """
    Function to train the netwok on a single epoch
    """

    loss_id = 0.0
    acc_hgr = 0.0
    total_samples = 0.0

    for batch_idx, dataSample in enumerate(tqdm.tqdm(dataloader,colour='blue')):
        
        x = dataSample['data'].to(device)
        y_id = dataSample['label'].to(device)
        
        model.train()
        #optimizer.zero_grad()

        with torch.set_grad_enabled(True):

            dense_id, f_theta = model.forward(x)
            loss_id_batch = obj_id(dense_id,y_id) # ID Loss
            loss_id_batch.backward()
            optimizer.step()
            optimizer.zero_grad()

        loss_id = loss_id + loss_id_batch.detach().item()*x.size(0)
        acc_id = acc_id + (torch.sum(y_id == torch.argmax(dense_id,dim=-1))).detach().item()
        total_samples = total_samples + x.size(0)

    return loss_id/total_samples, acc_id/total_samples

def val_epoch(dataloader, 
              model,  
              obj_id,
              device):
    
    """
    Function to test the netwok on a single epoch
    """

    loss_id = 0.0
    acc_hgr = 0.0
    total_samples = 0.0

    for batch_idx, dataSample in enumerate(tqdm.tqdm(dataloader,colour='green')):
        
        x = dataSample['data'].to(device)
        y_id = dataSample['label'].to(device)

        model.eval()
        with torch.set_grad_enabled(False):

            dense_id, f_theta = model.forward(x)
            loss_id_batch = obj_id(dense_id,y_id) # ID Loss

        loss_id = loss_id + loss_id_batch.detach().item()*x.size(0)
        acc_id = acc_id + (torch.sum(y_id == torch.argmax(dense_id,dim=-1))).detach().item()
        total_samples = total_samples + x.size(0)

    return loss_id/total_samples, acc_id/total_samples

def trainVal(train_loader,
              val_loader,
              model, 
              optimizer, 
              obj_id,
              args):
    
    """
    Function to train and validate in a single GPU setting
    """

    model_path = './_store/_weights/'+args.exp_name+'.pth'
    loss_best = 1e+6
    device = torch.device(args.device)

    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []
    train_metrics = []
    val_metrics = []

    for epoch in range(args.num_epochs):

        time_start = time.time()
        print(f'Epoch {epoch+1}/{args.num_epochs}')
        print('-' * 10)

        ##### Training
        loss_id_train_curr, acc_id_train_curr = train_epoch(train_loader,
                                                            model,
                                                            optimizer,
                                                            obj_id,
                                                            device)

        loss_train.append(loss_id_train_curr)
        acc_train.append(acc_id_train_curr)

        ##### Validation
        loss_id_val_curr, acc_id_val_curr = val_epoch(val_loader,
                                                      model,
                                                      obj_id,
                                                      device)
        
        if(loss_id_val_curr < loss_best):
            loss_best = loss_id_val_curr
            torch.save(model.state_dict(),model_path)

        loss_val.append(loss_id_train_curr)
        acc_val.append(acc_id_train_curr)

        ##### Outputs
        print('Total time:'+str(time.time() - time_start))
        print('Loss: '+str(loss_id_train_curr))
        print('Validation Loss: '+str(loss_id_val_curr))
        print('ID Accuracy: '+str(acc_id_train_curr))
        print('Validation ID Accuracy: '+str(acc_id_val_curr))

        wandb.log({'epoch':epoch,
                   'loss_id':loss_id_train_curr,
                   'acc_id':acc_id_train_curr,
                   'val_loss_id':loss_id_val_curr,
                   'val_acc_id':acc_id_val_curr,
                   })

    ##### Storing
    train_metrics.append(loss_train)
    train_metrics.append(acc_train)

    val_metrics.append(loss_val)
    val_metrics.append(acc_val)

    return train_metrics, val_metrics

