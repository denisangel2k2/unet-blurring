import logging
from matplotlib import pyplot as plt
import numpy as np
import torch
import wandb
from dataset.LFWDataset import *
from torchvision.transforms import v2
from evaluation import eval_model
from model.UNET import *



def train_model(
        dataset,
        model,
        device,
        epochs=5,
        learning_rate=0.001,
        weight_decay=1e-8,
        save_checkpoint=False,
        momentum=0.999,
        validation_percent=0.2,
        test_percent=0.1
):
    model.to(device)
    #Split into train test
    n_test=int(len(dataset)*test_percent)
    n_validation=int(len(dataset)*validation_percent)
    n_train=len(dataset)-n_test-n_validation

    train_dataset,test_dataset,validation_dataset=torch.utils.data.random_split(dataset,[n_train,n_test,n_validation])
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=4,shuffle=False)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=4,shuffle=False)
    validation_loader=torch.utils.data.DataLoader(validation_dataset,batch_size=4,shuffle=False)



    #Initialize the wandb project
    experiment=wandb.init(project='UNET2',resume='allow')
    experiment.config.update(
        dict(epochs=epochs, batch_size=4, learning_rate=learning_rate, test_percent=test_percent,validation_percent=validation_percent,save_checkpoint=save_checkpoint)
    )
    

    wandb.run.summary['best_mpa']=0

    logging.info(f'''Train starting with params:
            Epochs: {epochs}
            Batch size: {4}
            Learning rate: {learning_rate}
            Test size: {n_test},
            Device: {device},
            Save checkpoint: {save_checkpoint}
    ''')
    logging.basicConfig(level=logging.ERROR)
    #Optimizer and criterion
    # optimizer = torch.optim.RMSprop(model.parameters(),lr=learning_rate,weight_decay=weight_decay, momentum=momentum)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    criterion= torch.nn.CrossEntropyLoss()
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=2)

    for epoch in range(1,epochs+1):
        model.train()
        epoch_loss=0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images=batch[0]
                masks=batch[1]

                images=images.to(device,dtype=torch.float32)
                masks=masks.to(device)


                masks_predictions=model(images)
                loss=criterion(masks_predictions,masks)
                epoch_loss+=loss.item()*images.size(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])

                experiment.log(
                    {
                        'train loss': loss.item(),
                        'epoch': epoch,
                    }
                )


                pbar.set_postfix(**{'loss (batch)': loss.item()})   
            scheduler.step(epoch_loss/n_train)
            mpa=eval_model(model,validation_loader,experiment,criterion,device,epoch)

            #if mpa is better than the best_mpa then we save the wandb artifact
            if mpa>wandb.run.summary['best_mpa']:
                experiment.summary['best_mpa']=mpa
                torch.save(model.state_dict(), 'D:\\UBB\\CVDL\\Project\\checkpoints\\best_run.pth')
                artifact=wandb.Artifact('best_model',type='model', metadata={'mpa': mpa})
                artifact.add_file('D:\\UBB\\CVDL\\Project\\checkpoints\\best_run.pth')
                wandb.run.log_artifact(artifact) 

            experiment.log({'epoch loss': epoch_loss/n_train})

def train_sweep( 
 ):
    
    dataset=LFWDataset(download=False, base_folder='D:\\UBB\\CVDL\\Project\\dataset\\lfw_dataset', transforms=v2.Compose([
                                                            v2.Resize(256),
                                                            v2.CenterCrop(224),
                                                            v2.ToTensor()]))
    model = UNet(n_channels=3,n_classes=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    experiment = wandb.init(project='UNET2', resume='allow')

    config=wandb.config
    # Extract hyperparameters from the config
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    weight_decay = 1e-8
    momentum = config['momentum']
    validation_percent = config['validation_percent']
    test_percent = config['test_percent']

    experiment.config.update(
        dict(epochs=epochs, batch_size=4, learning_rate=learning_rate, 
             test_percent=test_percent, validation_percent=validation_percent, 
             save_checkpoint=False)
    )


    n_test=int(len(dataset)*test_percent)
    n_validation=int(len(dataset)*validation_percent)
    n_train=len(dataset)-n_test-n_validation

    train_dataset,test_dataset,validation_dataset=torch.utils.data.random_split(dataset,[n_train,n_test,n_validation])
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=4,shuffle=False)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=4,shuffle=False)
    validation_loader=torch.utils.data.DataLoader(validation_dataset,batch_size=4,shuffle=False)

    wandb.run.summary['best_mpa']=0

    logging.info(f'''Train starting with params:
            Epochs: {epochs}
            Batch size: {4}
            Learning rate: {learning_rate}
            Test size: {n_test},
            Device: {device},
            Save checkpoint: {False}
    ''')
    logging.basicConfig(level=logging.ERROR)



    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch[0]
                masks = batch[1]

                images = images.to(device, dtype=torch.float32)
                masks = masks.to(device)

                masks_predictions = model(images)
                loss = criterion(masks_predictions, masks)
                epoch_loss += loss.item() * images.size(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])

                experiment.log(
                    {
                        'train loss': loss.item(),
                        'epoch': epoch,
                    }
                )

                pbar.set_postfix(**{'loss (batch)': loss.item()})

            scheduler.step(epoch_loss / n_train)
            mpa = eval_model(model, validation_loader, experiment, criterion, device, epoch)

            # if mpa is better than the best_mpa then we save the wandb artifact
            if mpa > wandb.run.summary['best_mpa']:
                experiment.summary['best_mpa'] = mpa
                torch.save(model.state_dict(), 'D:\\UBB\\CVDL\\Project\\checkpoints\\best_run.pth')
                artifact = wandb.Artifact('best_model', type='model', metadata={'mpa': mpa})
                artifact.add_file('D:\\UBB\\CVDL\\Project\\checkpoints\\best_run.pth')
                wandb.run.log_artifact(artifact)

            experiment.log({'epoch loss': epoch_loss / n_train})                   

def main():
    
    # dataset=LFWDataset(download=False, base_folder='D:\\UBB\\CVDL\\Project\\dataset\\lfw_dataset', transforms=v2.Compose([
    #                                                         v2.Resize(256),
    #                                                         v2.CenterCrop(224),
    #                                                         v2.ToTensor()]))
    # model = UNet(n_channels=3,n_classes=3)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # #train_model(dataset,model,device,epochs=1)
    # torch.save(model.state_dict(), 'D:\\UBB\\CVDL\\Project\\model\\model9.pth')


    sweep_config = {
    'method': 'random',
    'name': 'your_sweep_name',
    'metric': {
        'name': 'best_mpa',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'values': [3]
        },
        'learning_rate': {
            'min': 0.0001,
            'max': 0.01,
            'distribution': 'uniform'
        },
        'momentum': {
            'min': 0.9,
            'max': 0.999,
            'distribution': 'uniform'
        },
        'validation_percent': {
            'min': 0.1,
            'max': 0.3,
            'distribution': 'uniform'
        },
        'test_percent': {
            'min': 0.05,
            'max': 0.2,
            'distribution': 'uniform'
        }
    }
}

    sweep_id = wandb.sweep(sweep_config, project='UNET2')
    wandb.agent(sweep_id,function=train_sweep,count=2)


    
main()



