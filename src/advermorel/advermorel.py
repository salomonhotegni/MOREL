"""Main module."""

import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from advermorel.models.morelnet import morelnet
from advermorel.losses.morel import morel_loss

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

import datetime
from time import time, sleep
from tqdm import tqdm

class MOREL:
    def __init__(self, original_model, 
                 name_last_layer,
                 num_class,
                 device,
                 accu_obj="mart",
                 k=[0.1, 0.9],
                 a=[0.0]*2,
                 gamma=2e-5,
                 alpha=1e-5,
                 embed_dim=128,
                 num_att_heads=2,
                 dropout=0.0,
                 epsilon=0.031,
                 train_step_size=0.007,
                 eval_step_size=0.003,
                 train_perturb_steps=10,
                 eval_perturb_steps=20,
                 beta=6.0,
                 distance="l_inf",
                 loat_lore_type="LORE_v1",
                 loat_reg=True,
                 loat_reg_type="mse",
                 loat_from_epoch=1,
                 loat_adv=True,
                 loat_gamma =0.05,
                 loat_theta=1.0,
                 custom_accu_obj=None,
                 model_dir="logs/",
                 verbose=True):
        """
        The MOREL class implements the MOREL adversarial training framework.
        Args:
            original_model: The original model to be trained.
            name_last_layer: The name of the last layer in the model.
            num_class: Number of classes in the dataset.
            device: The device to run the model on.
            accu_obj: The accuracy loss function (e.g., "trades", "mart", "loat").
            k: Preference vector for the MOREL loss.
            a: Reference point for the MOREL loss.
            gamma: Augmentation coefficient for the MOREL loss.
            alpha: Weight for the Contrastive loss.
            embed_dim: Embedding dimension for the model.
            num_att_heads: Number of attention heads in the model.
            dropout: Dropout rate for the model.
            epsilon: Maximum perturbation for adversarial examples.
            train_step_size: Step size for adversarial perturbation during training.
            eval_step_size: Step size for adversarial perturbation during evaluation.
            train_perturb_steps: Number of perturbation steps during training.
            eval_perturb_steps: Number of perturbation steps during evaluation.
            beta: Weight for the robustness loss.
            distance: Distance metric for perturbation ("l_inf" or "l_2").
            loat_lore_type: Type of LORE to use for LoAT.
            loat_reg: Whether to use regularization in LoAT.
            loat_reg_type: Type of regularization to use in LoAT.
            loat_from_epoch: Epoch from which to start using LoAT.
            loat_adv: Whether to use adversarial examples in LoAT.
            loat_gamma: Gamma parameter for LoAT.
            loat_theta: Theta parameter for LoAT.
            custom_accu_obj: Custom accuracy loss function if provided. It should accept the following parameters:
                model, x_natural, y, optimizer, and extra_configs, where extra_configs is a dictionary
                containing any additional configurations needed by the custom loss function.
                It should return a tuple of (loss for accuracy, adversarial batch).
            model_dir: Directory to save the model checkpoints.
            verbose: Whether to print verbose output during training.
        """

        self.num_class = num_class
        self.device = device
        self.accu_obj = accu_obj
        self.k = k
        self.a = a
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.train_step_size = train_step_size
        self.eval_step_size = eval_step_size
        self.train_perturb_steps = train_perturb_steps
        self.eval_perturb_steps = eval_perturb_steps
        self.beta = beta
        self.distance = distance
        self.custom_accu_obj = custom_accu_obj
        self.model_dir = model_dir
        self.verbose = verbose
        if "loat" in accu_obj:
            self.loat_lore_type = loat_lore_type  # LORE type
            self.loat_reg = loat_reg  # regularization flag
            self.loat_reg_type = loat_reg_type  # regularization type
            self.loat_from_epoch = loat_from_epoch
            self.loat_adv = loat_adv
            self.loat_gamma = loat_gamma
            self.loat_theta = loat_theta
        
        if not os.path.exists("%s" % (self.model_dir)):
            os.makedirs("%s" % (self.model_dir))
        
        mod_encoder = original_model
        head_classifier = getattr(mod_encoder, name_last_layer)
        self.feat_dim = head_classifier.in_features
        setattr(mod_encoder, name_last_layer, nn.Identity()) # Remove classifier from Encoder
        kwargs = {
            "embed_dim": embed_dim,
            "num_att_heads": num_att_heads,
            "dropout": dropout,
        }
        self.model = morelnet(mod_encoder, head_classifier, **kwargs)

    def get_morel_loss(self, X, y, optimizer,
                       epoch, extra_configs=None):
        """
        Computes the MOREL loss for a given batch.
        Args:
            X: Input data.
            y: Target labels.
            optimizer: Optimizer for the model.
            epoch: Current epoch number.
            extra_configs: Additional configurations for the loss function.
        Returns:
            ConeScal_loss: The computed MOREL loss.
            orig_loss: Original losses (robust loss and clean loss).
        """
        if self.accu_obj is None:
            return morel_loss(
                    model=self.model,
                    x_natural=X,
                    y=y,
                    optimizer=optimizer,
                    k=self.k,
                    a=self.a,
                    gamma=self.gamma,
                    accu_obj=self.accu_obj,
                    alpha=self.alpha,
                    extra_configs=extra_configs,
                    custom_accu_obj=self.custom_accu_obj,
                )
        elif "trades" in self.accu_obj or "mart" in self.accu_obj:
                return morel_loss(
                model=self.model,
                x_natural=X,
                y=y,
                optimizer=optimizer,
                k=self.k,
                a=self.a,
                gamma=self.gamma,
                accu_obj=self.accu_obj,
                alpha=self.alpha,
                step_size=self.train_step_size,
                epsilon=self.epsilon,
                perturb_steps=self.train_perturb_steps,
                beta=self.beta,
                distance=self.distance,
                extra_configs=None,
                extra_outputs=True,
            )
        elif "loat" in self.accu_obj:
                return morel_loss(
                    model=self.model,
                    x_natural=X,
                    y=y,
                    optimizer=optimizer,
                    k=self.k,
                    a=self.a,
                    gamma=self.gamma,
                    accu_obj=self.accu_obj,
                    alpha=self.alpha,
                    step_size=self.train_step_size,
                    epsilon=self.epsilon,
                    perturb_steps=self.train_perturb_steps,
                    beta=self.beta,
                    distance=self.distance,
                    extra_configs={"epoch": epoch,
                                "reg": self.loat_reg,
                                "reg_type": self.loat_reg_type,
                                "device": self.device,
                                "theta": self.loat_theta,
                                "gamma_loat": self.loat_gamma,
                                "lot": self.loat_lore_type,
                    },
                    extra_outputs=True,
                )
        else:
            raise ValueError(f"Loss function {self.accu_obj} not implemented !")

    def train(self, 
              optimizer, 
              num_epochs, train_loader, 
              val_loader=None, scheduler=None,
              val_attack=None, resume_train=False, 
              extra_configs=None, seed=0):
        """
        Train the model with MOREL.
        Args:
            optimizer: The optimizer to use for training.
            num_epochs: The number of epochs to train for.
            train_loader: The data loader for the training set.
            val_loader: The data loader for the validation set (optional).
            scheduler: The learning rate scheduler (optional).
            val_attack: The adversarial attack to use during validation (optional).
            resume_train: Whether to resume training from a checkpoint (optional).
            extra_configs: Additional configurations for the loss function (optional).
            seed: Random seed for reproducibility (default: 0).
        """
        print("#############################################")
        print("######### Training model with MOREL #########")
        print("#############################################")
        
        print("Model will be saved at: ", self.model_dir)
        print(f"Preference vector: {self.k}")
        if self.accu_obj is not None:
            print(f"Loss function for accuracy: {self.accu_obj.upper()}")
        elif self.custom_accu_obj is None:
            raise ValueError("Either accu_obj or custom_accu_obj must be provided.")

        self.set_seed(seed) 
        self.model = self.model.to(self.device) # morel_net.to(self.device)
        
        self.model.train()
        
        ALL_TRAIN_LOSS = []
        ALL_VAL_ACCU = []
        ALL_ORIG_LOSS = []
        last_best_val_accu_epch = None
        best_val_accu = 0.0
        best_val_adv_accu = 0.0
        best_val_clean_accu = 0.0
            
        init_epoch = 0
        if resume_train:
            (
                last_epoch,
                ALL_TRAIN_LOSS,
                ALL_VAL_ACCU,
                ALL_ORIG_LOSS,
                last_best_val_accu_epch,
            ) = self.load_checkpoint(self.model, optimizer, scheduler, self.model_dir)
            if last_best_val_accu_epch is not None:
                best_val_adv_accu, best_val_clean_accu, best_adv_epch, best_clean_epch = (
                    last_best_val_accu_epch
                )
                if self.verbose:
                    print("================================================================")
                    print(f"Best Robust Accuracy so far: {best_val_adv_accu} at epoch {best_adv_epch}.")
                    print(f"Best Clean Accuracy so far: {best_val_clean_accu} at epoch {best_clean_epch}.")
                    print("================================================================")
            init_epoch = last_epoch + 1
            print(f"*** Resuming training from epoch {init_epoch} ***")
        
        for epoch in range(init_epoch, num_epochs):
            t0_epch = time()
            print("****************************")
            print(f"******* EPOCH {epoch} / {num_epochs} ********")
            print("****************************")
            train_loss, orig_train_losses = self.train_epoch(
                train_loader, optimizer, epoch, extra_configs=extra_configs
            )
            ALL_TRAIN_LOSS.append(train_loss)
            ALL_ORIG_LOSS.append(orig_train_losses.numpy())
            if scheduler is not None:
                scheduler.step()
            if val_loader is not None:
                accuracy_clean, accuracy_robust = self.evaluate_epoch(val_loader, attack=val_attack)
                
                ALL_VAL_ACCU.append([accuracy_robust, accuracy_clean])
                gen_val_accuracy = (accuracy_robust + accuracy_clean) / 2
                
                if gen_val_accuracy >= best_val_accu:
                    best_val_accu = gen_val_accuracy
                    
                if accuracy_clean >= best_val_clean_accu:
                    if self.verbose: print("* Best Clean Accuracy !")
                    best_val_clean_accu = accuracy_clean

                if accuracy_robust >= best_val_adv_accu:
                    if self.verbose: print("* Best Robust Accuracy !")
                    best_val_adv_accu = accuracy_robust
                    last_best_val_accu_epch = [
                        best_val_accu,
                        best_val_adv_accu,
                        best_val_clean_accu,
                        epoch,
                    ]
                    # This is denoted as 'best' in the paper
                    self.save_checkpoint(
                        epoch,
                        optimizer,
                        scheduler,
                        ALL_TRAIN_LOSS,
                        ALL_VAL_ACCU,
                        ALL_ORIG_LOSS=ALL_ORIG_LOSS,
                        best_val_accuracy_epch=last_best_val_accu_epch,
                        saving_best=True,
                    )
                if self.verbose:
                    print(
                        "Average Accuracy: {:.2f}%    (Best: {:.2f}%)".format(
                            gen_val_accuracy, best_val_accu
                        )
                    )
                    print("===========================")

            # Save the actual training state
            self.save_checkpoint(
                epoch,
                optimizer,
                scheduler,
                ALL_TRAIN_LOSS,
                ALL_VAL_ACCU,
                ALL_ORIG_LOSS=ALL_ORIG_LOSS,
                best_val_accuracy_epch=last_best_val_accu_epch,
                saving_best=False,
            )
            T_epch = time() - t0_epch
            # Print computation time
            print("Time: {} minutes".format(T_epch / 60))
            print(datetime.datetime.now())
        
    def train_epoch(self, train_loader, optimizer, 
                    actu_epoch, extra_configs=None):
        """
        Train the model for one epoch using the MOREL loss.
        Args:
            train_loader: DataLoader for the training data.
            optimizer: Optimizer for the model.
            actu_epoch: Current epoch number.
            extra_configs: Additional configurations for the loss function.
        Returns:
            train_loss: Average training loss for the epoch.
            orig_train_losses: Original losses (robust loss and clean loss) for the epoch.
        """
        self.model.train()
        pbar = tqdm(total=len(train_loader), 
                        desc="Training", unit="Batch")
        train_loss = 0.0
        orig_train_losses = torch.zeros([2, 1])
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            loss, orig_loss =  self.get_morel_loss(data, target,
                                                    optimizer, actu_epoch,
                                                    extra_configs=extra_configs)  
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.item()
            orig_train_losses += orig_loss
            
            pbar.set_postfix({"Loss": loss.item()})
            pbar.update(1)

        train_loss = train_loss / len(train_loader)
        orig_train_losses = orig_train_losses / len(train_loader)
        return train_loss, orig_train_losses

    def evaluate_epoch(self, val_loader, attack=None):
        """
        Evaluate clean and adversarial accuracy using an ART attack.
        
        Args:
            val_loader  : DataLoader for validation set.
            attack      : an ART Attack object (must implement .generate()).
                          If None, PGD-20 will be used.
                          
        Returns:
            (accuracy_clean, accuracy_robust) in percentages.
        """
        default_optimizer = optim.SGD(
                            self.model.parameters(),
                            lr=0.01,
                        )
        self.model.eval()
        total = len(val_loader.dataset)
        clean_errors = 0
        robust_errors = 0

        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)

            # 1) Natural error
            with torch.no_grad():
                logits = self.model(data)
                preds = logits.argmax(dim=1)
                clean_errors += (preds != target).sum().item()

            # 2) If no attack,use PGD-20
            if attack is None:
                classifier_att = PyTorchClassifier(
                    model=self.model,
                    clip_values=(0.0, 1.0),
                    loss=nn.CrossEntropyLoss(),
                    optimizer=default_optimizer,
                    input_shape=data.shape[1:],
                    nb_classes=self.num_class,
                )
                attack = ProjectedGradientDescent(
                    estimator=classifier_att,
                    norm=np.inf,
                    eps=self.epsilon,
                    eps_step=self.eval_step_size,
                    max_iter=self.eval_perturb_steps,
                    targeted=False,
                    num_random_init=0,
                    batch_size=data.shape[0],
                )

            # 3) Generate adversarial examples via ART
            #    ART attacks expect numpy arrays of shape (N, C, H, W)
            x_np = data.detach().cpu().numpy()
            y_np = target.detach().cpu().numpy()
            x_adv_np = attack.generate(x_np, y=y_np)
            # Convert back to torch tensor on device
            x_adv = torch.from_numpy(x_adv_np).to(self.device)

            # 4) Robust error
            with torch.no_grad():
                adv_logits = self.model(x_adv)
                adv_preds = adv_logits.argmax(dim=1)
                robust_errors += (adv_preds != target).sum().item()

        acc_clean  = 100.0 * (1 - clean_errors  / total)
        acc_robust = 100.0 * (1 - robust_errors / total)
        if self.verbose:
            print("===== Validation set: =====")
            print(f"Clean Accuracy:  {acc_clean:.2f}%")
            print(f"Robust Accuracy: {acc_robust:.2f}%")
        return acc_clean, acc_robust

    def test(self, 
             test_loader,
             optimizer = None,
             attack=None,
             best_model=False):
        """
        Test the model on the given test set using the specified attack.
        Args:
            test_loader: DataLoader for the test set.
            optimizer: Optimizer for the model (optional).
            attack: An ART Attack object (must implement .generate()).
            best_model: Whether to load the best model checkpoint (default: False).
        Returns:
            (accuracy_clean, accuracy_robust) in percentages.
        """
        
        assert attack is not None, "Attack must be provided for testing."

        self.model.eval()
        correct_clean = 0
        correct_adv = 0
        if optimizer is None:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=0.01,
            )

        if best_model:
            self.load_checkpoint(loading_best=True)
        else:
            self.load_checkpoint(loading_best=False)
            
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)

            with torch.enable_grad():
                data_attack = attack.generate(data.cpu().numpy(), y=target.cpu().numpy())
                data_attack = torch.from_numpy(data_attack).to(self.device)
            with torch.no_grad():
                # Forward pass on clean data
                output = self.model(data)
                output = F.log_softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)
                is_correct_clean = pred.eq(target.view_as(pred)).sum().item()
                correct_clean += is_correct_clean
                # Forward pass on adversarial data
                output = self.model(data_attack)
                output = F.log_softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)
                is_correct_adv = pred.eq(target.view_as(pred)).sum().item()
                correct_adv += is_correct_adv

        accuracy_clean = 100.0 * correct_clean / len(test_loader.dataset)
        accuracy_robust = 100.0 * correct_adv / len(test_loader.dataset)
        
        print("Clean Accuracy: ", accuracy_clean)
        print("Robust Accuracy: ", accuracy_robust)

        return accuracy_clean, accuracy_robust
  
    def load_checkpoint(
        self,
        optimizer=None,
        scheduler=None,
        loading_best=False,
    ):
        """ Load the model checkpoint."""
        
        if loading_best:
            filename = str("%s/best_train_checkpoint.pth" % (self.model_dir))
        else:
            filename = str("%s/last_train_checkpoint.pth" % (self.model_dir))

        checkpoint = torch.load(filename, weights_only=False, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint["epoch"]
        last_best_val_accu_epch = checkpoint["last_best_valid_accuracy_epoch"]
        all_train_loss = checkpoint["list_train_loss"]
        all_val_accu = checkpoint["list_val_accuracy"]
        all_orig_loss = checkpoint["list_orig_loss"]

        if self.verbose:
            which_model = "Best" if loading_best else "Last"
            print(f"{which_model} checkpoint loaded.")

        return (
            epoch,
            all_train_loss,
            all_val_accu,
            all_orig_loss,
            last_best_val_accu_epch,
        )
        
    def save_checkpoint(
        self,
        epoch,
        optimizer,
        scheduler,
        ALL_TRAIN_LOSS,
        ALL_VAL_ACCU,
        ALL_ORIG_LOSS,
        best_val_accuracy_epch=None,
        saving_best=False,
    ):
        """ Save the model checkpoint."""
        
        if not os.path.exists("%s" % (self.model_dir)):
            os.makedirs("%s" % (self.model_dir))
        TRAIN_STATE_FILE = str("%s/last_train_checkpoint.pth" % (self.model_dir))
        BESTTRAIN_STATE_FILE = str("%s/best_train_checkpoint.pth" % (self.model_dir))

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "last_best_valid_accuracy_epoch": best_val_accuracy_epch,
            "list_train_loss": ALL_TRAIN_LOSS,
            "list_val_accuracy": ALL_VAL_ACCU,
            "list_orig_loss": ALL_ORIG_LOSS,
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if saving_best:
            torch.save(checkpoint, BESTTRAIN_STATE_FILE)
        else:
            torch.save(checkpoint, TRAIN_STATE_FILE)

    def set_seed(seed: int = 0):
        """
        Set the random seed for reproducibility.
        """
        np.random.seed(seed)
        random.seed(seed)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True       