from torchvision import transforms
from filelock import FileLock
from torch.utils.data import DataLoader, random_split
from FBMDataset import FBMDataset
import torch
import os
from datetime import datetime
import torch.nn.functional as F
from torch.optim import SGD
from FBMClassifier import FBMClassifier
from ray import tune
from ray.air import session,Checkpoint
from torch.utils.tensorboard import SummaryWriter

class FBMTrainer:   
    '''A class to perform the training of the model for the facebook marketplace classification problem.

    Methods
    ----------
    get_datasets()
            Function to retrieve, shuffle and split the datasets.
    create_model_dir_path()
            Function to create directory path to store model information per run.
    training_loop()
            Representation of a training run for a single epoch of the resnet50 model.
    validate()
            Representation of validation for a single epoch of the resnet50 model.
    test()
            Perform testing after all epochs have been run on a single solution.
    train_fbm()
            Main function to perform full run of the training and validation of the model over multiple epochs. Also used for hyperparameter tuning.
    '''
    def get_datasets(self, training_data_dir, cleaned_images_dir):    
        transform_list = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225])
            ])
        with FileLock(os.path.expanduser("~/.data.lock")):
            dataset = FBMDataset(training_data_dir,cleaned_images_dir,transform=transform_list)
            #obtain the list of targets
            train_dataset,test_dataset,val_dataset = random_split(dataset, [0.7,0.2,0.1])
        return (train_dataset, test_dataset, val_dataset)
    
    def create_model_dir_path(self):
        parent_dir = 'model_evaluation'
        current_datetime = datetime.now().strftime('%y%m%d%H%M%S')
        child_dir = 'weights'
        path = os.path.join(os.getcwd(), parent_dir, current_datetime, child_dir)
        return path

    def training_loop(self, model, optimiser, train_loader, epoch_num,device=None):  
        writer = SummaryWriter()
        batch_id = 0     
        # Set the model to run on the device
        model = model.to(device)
        model.train(True)     
        print(f'Beginning Batches for epoch {epoch_num}')
        print(len(train_loader))
        for batch in train_loader:
            # get features and labels from the batch
            features,labels = batch
            features = features.to(device)
            labels = labels.to(device, non_blocking=True)
            # loss.backward does not overwrite, it adds. To stop this, we set the gradients back to zero. sets the .grad of all optimized tensors to zero
            optimiser.zero_grad()
            # make a prediction
            prediction = model(features)
            # calculate loss
            criterion = F.cross_entropy(prediction,labels)
            # backward function calculates the gradient of the current tensor w.r.t graph leaves
            criterion.backward()
            # moves each parameter in the opposite direction of the gradient, proportional to the learning rate
            optimiser.step()
            writer.add_scalar('Loss', criterion.item(), batch_id)
            batch_id += 1
            print(f"completed: {batch_id}")
        print("Completed")
        

    def validate(self, model, val_loader, device):
        # Set the model to evaluation mode
        model.eval()
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        return (val_loss / val_steps),(correct / total)
    
    def test(self, model, test_loader, device):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                features, labels = data
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
    
    def train_fbm(self, config):    
        hyperparameter_tuning_on=config["hyperparameter_tuning_on"]
        is_feature_extraction_model=config["is_feature_extraction_model"]
        model = FBMClassifier(is_feature_extraction_model)
        optimiser = SGD(model.resnet50.parameters(), lr = config["lr"], momentum=0.9)     
        base_dir = "D:/Documents/AICore/facebook-marketplaces-recommendation-ranking-system"
        train_dataset, test_dataset, val_dataset = self.get_datasets(os.path.join(base_dir,"data/training_data.csv"),os.path.join(base_dir,"cleaned_images"))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        # Set the model to run on the device
        model = model.to(device)

        train_loader = DataLoader(train_dataset,batch_size=config["batch_size"],shuffle=True)
        test_loader = DataLoader(test_dataset,batch_size=config["batch_size"],shuffle=True)
        val_loader = DataLoader(val_dataset,batch_size=config["batch_size"],shuffle=True)
        
        path = self.create_model_dir_path()
        os.makedirs(path)
        
        for epoch in range(1):
            print(f"Beginning {epoch} ...")
            self.training_loop(model,optimiser,train_loader, epoch, device=device)
            print('Training complete ...')
            loss, accuracy = self.validate(model, val_loader, device=device)
            print('Validation complete ...')        
            print(f'Epoch {epoch} - Average Loss: {loss}')
            print(f'Epoch {epoch} - Accuracy: {accuracy}')

            if hyperparameter_tuning_on:
                os.makedirs("my_model", exist_ok=True)
                torch.save(
                    (model.state_dict(), optimiser.state_dict()), "my_model/checkpoint.pt")
                checkpoint = Checkpoint.from_directory("my_model")
                session.report({"loss": loss, "accuracy": accuracy}, checkpoint=checkpoint)
            elif is_feature_extraction_model:
                torch.save(model.state_dict(), path + f'/image_model.pt')
            else:    
                torch.save(model.state_dict(), path + f'/epoch_{epoch}.pt')
            print(f"Ending {epoch} ...")    

            test_accuracy = self.test(model,test_loader, device)
            print(f"Testing Accuracy: {test_accuracy}")
            return test_accuracy
        
if __name__ == '__main__':
    trainer = FBMTrainer()
    config = {        
            "lr": tune.loguniform(1e-2,1e-1),
            "batch_size": tune.choice([8]),
            "hyperparameter_tuning_on": True,
            "is_feature_extraction_model": False
        }
    tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(trainer.train_fbm),
                resources=tune.PlacementGroupFactory([{"CPU": 2,"GPU": 1}])
            ),
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                num_samples=1,
            ),
            param_space=config
        )

    results = tuner.fit()
    best_result = results.get_best_result("loss", "min")