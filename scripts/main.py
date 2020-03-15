import numpy as np
import torch
import torch.optim as optim
from script_test_secML_attack_on_Keras_model import transform_dataset, attack_keras_model
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from tqdm import trange
from tqdm.notebook import tnrange
from torch.utils.data.dataset import ConcatDataset

from sklearn import preprocessing
import pandas as pd
import os
import argparse
import logging
from torch.autograd import Function
import matplotlib.pyplot as plt
from bayesian_model import BayesianModel as bm
from pycm import ConfusionMatrix

from secml.array.c_array import CArray

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.DEBUG)


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class Net(nn.Module):

    def __init__(self, grl_lambda=100):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(14, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.grl = GradientReversal(grl_lambda )
        self.fc4 = nn.Linear(64, 2)

        # self.grl = GradientReversal(100)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.1)
        x = self.fc2(x)
        x = F.relu(x)
        hidden = F.dropout(x, 0.1)
        y = self.fc3(hidden)
        y = F.dropout(y, 0.1)
        s = self.grl(hidden)
        s = self.fc4(s)
        # s = F.sigmoid(s)
        s = F.dropout(s, 0.1)
        return y, s


def train_and_evaluate(train_loader: DataLoader,
                       val_loader: DataLoader,
                       test_loader: DataLoader,
                       device,
                       args,
                       grl_lambda = None):
    """

    :param train_loader: Pytorch-like DataLoader with training data.
    :param val_loader: Pytorch-like DataLoader with validation data.
    :param test_loader: Pytorch-like DataLoader with testing data.
    :param device: The target device for the training.
    :return: A tuple: (trained Pytorch-like model, dataframe with results on test set)
    """

    torch.manual_seed(0)

    grl_lambda = grl_lambda if grl_lambda is not None else args.grl_lambda

    model = Net(grl_lambda).to(device)
    criterion = nn.MSELoss().to(device)
    criterion_bias = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adagrad(model.parameters())

    training_losses = []
    validation_losses = []

    t_prog = trange(args.epochs, desc='Training neural network', leave=False, position=1, mininterval=5)
    # t_prog = trange(50)

    for epoch in t_prog:
        model.train()

        batch_losses = []
        for x_batch, y_batch, _, s_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            s_batch = s_batch.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, outputs_protected = model(x_batch)
            if grl_lambda is not None:
                loss = criterion(outputs, y_batch) + criterion_bias(outputs_protected, s_batch.argmax(dim=1))
            else:
                loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)

        with torch.no_grad():
            val_losses = []
            for x_val, y_val, _, s_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                s_val = s_val.to(device)
                model.eval()
                yhat, s_hat = model(x_val)
                if grl_lambda is not None:
                    val_loss = (criterion(y_val, yhat) + criterion_bias(s_val, s_hat.argmax(dim=1))).item()
                else:
                    val_loss = criterion(y_val, yhat).item()
                val_losses.append(val_loss)
            validation_loss = np.mean(val_losses)
            validation_losses.append(validation_loss)

        t_prog.set_postfix({"epoch": epoch, "training_loss": training_loss,
                            "validation_loss": validation_loss}, refresh=False)  # print last metrics


    if args.show_graphs:
        plt.plot(range(len(training_losses)), training_losses)
        plt.plot(range(len(validation_losses)), validation_losses)
        # plt.scatter(x_tensor, y_out.detach().numpy())
        plt.ylabel('some numbers')
        plt.show()

    with torch.no_grad():
        test_losses = []
        test_results = []
        for x_test, y_test, ytrue, s_true in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            s_true = s_true.to(device)
            model.eval()
            yhat, s_hat = model(x_test)
            test_loss = (criterion(y_test, yhat) + criterion_bias(s_true, s_hat.argmax(dim=1))).item()
            test_losses.append(val_loss)
            test_results.append({"y_hat": yhat, "y_true": ytrue, "y_compas": y_test, "s": s_true, "s_hat": s_hat})

        #print({"Test loss": np.mean(test_losses)})

    results = test_results[0]['y_hat']
    outcome = test_results[0]['y_true']
    compas = test_results[0]['y_compas']
    protected_results = test_results[0]['s']
    protected = test_results[0]['s_hat']
    for r in test_results[1:]:
        results = torch.cat((results, r['y_hat']))
        outcome = torch.cat((outcome, r['y_true']))
        compas = torch.cat((compas, r['y_compas']))
        protected_results = torch.cat((protected_results, r['s']))
        protected = torch.cat((protected, r['s_hat']))

    df = pd.DataFrame(data=results.cpu().numpy(), columns=['pred'])

    df['true'] = outcome.cpu().numpy()
    df['compas'] = compas.cpu().numpy()
    df['race'] = protected_results.cpu().numpy()[:, 0]
    df['race_hat'] = protected.cpu().numpy()[:, 0]

    return model, df

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.debug("using device {} for pytorch.".format(device))

    df = pd.read_csv(os.path.join("..", "data", "csv", "scikit",
                                  "compas_recidive_two_years_sanitize_age_category_jail_time_decile_score.csv"))

    df_binary, Y, S, Y_true = transform_dataset(df)

    x_tensor = torch.tensor(df_binary.to_numpy().astype(np.float32))
    y_tensor = torch.tensor(Y.to_numpy().reshape(-1, 1).astype(np.float32))
    l_tensor = torch.tensor(Y_true.to_numpy().reshape(-1, 1).astype(np.float32))
    s_tensor = torch.tensor(preprocessing.OneHotEncoder().fit_transform(np.array(S).reshape(-1, 1)).toarray())

    dataset = TensorDataset(x_tensor, y_tensor, l_tensor, s_tensor)  # dataset = CustomDataset(x_tensor, y_tensor)

    base_size = len(dataset) // 10
    split = [7 * base_size, 1 * base_size, len(dataset) - 8 * base_size]  # Train, validation, test

    train_dataset, val_dataset, test_dataset = random_split(dataset, split)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)

    x_train_tensor = train_dataset[:][0]
    y_train_tensor = train_dataset[:][1]
    l_train_tensor = train_dataset[:][2]
    s_train_tensor = train_dataset[:][3]

    global_results = []

    _, results = train_and_evaluate(train_loader, val_loader, test_loader, device, args, grl_lambda=0)

    # Calculate biases after training
    dem_parity = abs(
        bm(results).P(pred=lambda x: x > 4).given(race=0)
        - bm(results).P(pred=lambda x: x > 4).given(
            race=1))

    eq_op = abs(
        bm(results).P(pred=lambda x: x > 4).given(race=0, compas=True)
        - bm(results).P(pred=lambda x: x > 4).given(race=1, compas=True))

    dem_parity_ratio = abs(
        bm(results).P(pred=lambda x: x > 4).given(race=0)
        / bm(results).P(pred=lambda x: x > 4).given(
            race=1))

    cm = ConfusionMatrix(actual_vector=(results['true'] == True).values,
                         predict_vector=(results['pred'] > 4).values)

    result = {"DP": dem_parity, "EO": eq_op, "DP ratio": dem_parity_ratio, "acc": cm.Overall_ACC, "f1": cm.F1_Macro}

    global_results.append(result)

    t_main = trange(args.iterations, desc="Attack", leave=False, position=0)
    for i in t_main:
        # Train network
        _, results = train_and_evaluate(train_loader, val_loader, test_loader, device, args)

        # Calculate biases after training
        dem_parity = abs(
            bm(results).P(pred=lambda x: x > 4).given(race=0)
            - bm(results).P(pred=lambda x: x > 4).given(
                race=1))

        eq_op = abs(
            bm(results).P(pred=lambda x: x > 4).given(race=0, compas=True)
            - bm(results).P(pred=lambda x: x > 4).given(race=1, compas=True))

        dem_parity_ratio = abs(
            bm(results).P(pred=lambda x: x > 4).given(race=0)
            / bm(results).P(pred=lambda x: x > 4).given(
                race=1))


        cm = ConfusionMatrix(actual_vector=(results['true'] == True).values,
                              predict_vector=(results['pred'] > 4).values)

        result = {"DP": dem_parity, "EO": eq_op, "DP ratio": dem_parity_ratio, "acc": cm.Overall_ACC, "f1": cm.F1_Macro}
        t_main.set_postfix(result)

        global_results.append(result)

        # Attack
        result_pts, result_class, labels = attack_keras_model(
            CArray(x_train_tensor),
            Y=CArray((y_train_tensor[:, 0] > 4).int()),
            S=s_train_tensor,
            nb_attack=25)

        # incorporate adversarial points
        x_train_tensor = torch.cat((x_train_tensor, torch.tensor(result_pts.astype(np.float32)).clamp(0, 1)))
        y_train_tensor = torch.cat((y_train_tensor, torch.tensor(result_class.reshape(-1, 1).astype(np.float32)).clamp(0, 10)))
        l_train_tensor = torch.cat((l_train_tensor, torch.tensor(labels.tondarray().reshape(-1, 1).astype(np.float32))))
        s = np.random.randint(2, size=len(result_class))
        s_train_tensor = torch.cat((s_train_tensor,  torch.tensor(np.array([s, 1 - s]).T.astype(np.float64))))

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor, l_train_tensor, s_train_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        logging.debug("New training dataset has size {} (original {}).".format(len(train_loader), base_size*7))

        df = pd.DataFrame(global_results)

        print(df)

if __name__ == '__main__':
    # Define arguments for cli and run main function
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--iterations', help="Number of attack iterations", default=20, type=int)
    parser.add_argument('--batch-size', help="Size of each minibatch for the classifier", default=2048, type=int)
    parser.add_argument('--show-graphs', help="Shows graph of training, etc. if true.", default=True)
    parser.add_argument('--grl-lambda', help="Gradient reversal parameter.", default=1, type=int)
    args = parser.parse_args()
    main(args)
