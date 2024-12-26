from torch_geometric.data import DataLoader
import torch
from tqdm import trange, tqdm
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
import wandb
import random

from dataset import CrystDataset
from diffusion1 import CSPDiffusion


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_random_seed(17)

trainset = CrystDataset('train.csv', 'train_sym')
testset = CrystDataset('test.csv', 'test_sym')

train_batch_size = 256
test_batch_size = 256

train_loader = DataLoader(trainset, shuffle=True, batch_size=train_batch_size)
test_loader = DataLoader(testset, shuffle=False, batch_size=test_batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CSPDiffusion(device).to(device)
model = torch.compile(model, fullgraph=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.6, patience=30, min_lr=1e-4
)

matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

wandb.init(
    project="diff_csp++"
)

num_epochs = 600

for epoch in trange(num_epochs):
    train_loss = []
    res_log = {}
    model.train(True)
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        loss = model.training_step(batch, 0)
        train_loss.append(loss.data.to('cpu').numpy())

        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 0.4)
        optimizer.step()
        optimizer.zero_grad()

    sheduler.step(np.mean(train_loss))
    res_log['train_loss'] = np.mean(train_loss)

    if epoch and not epoch % 499:
        model.train(False)
        with torch.no_grad():
            frac_coords, num_atoms, atom_types, lattices, input_data_list = [], [], [], [], []
            for idx, batch in tqdm(enumerate(test_loader)):
                batch = batch.to(device)
                outputs, _ = model.sample(batch)

                frac_coords.append(outputs['frac_coords'].detach().cpu())
                num_atoms.append(outputs['num_atoms'].detach().cpu())
                atom_types.append(outputs['atom_types'].detach().cpu())
                lattices.append(outputs['lattices'].detach().cpu())
                input_data_list = input_data_list + batch.to_data_list()

            frac_coords = torch.cat(frac_coords, dim=0)
            num_atoms = torch.cat(num_atoms, dim=0)
            atom_types = torch.cat(atom_types, dim=0)
            lattices = torch.cat(lattices, dim=0)

            preds_list = []
            start_idx = 0
            for i in range(len(num_atoms)):
                cur_frac_coords = frac_coords.narrow(0, start_idx, num_atoms[i])
                cur_atom_types = atom_types.narrow(0, start_idx, num_atoms[i])
                preds_list.append(
                    Structure(lattice=lattices[i], species=cur_atom_types, coords=cur_frac_coords, coords_are_cartesian=False)
                )
                start_idx += num_atoms[i]

            input_list = []
            for struct in input_data_list:
                input_list.append(
                    Structure(
                        lattice=Lattice.from_parameters(*(struct.lengths.tolist()[0] + struct.angles.tolist()[0])),
                        species=struct.atom_types.to('cpu'),
                        coords=struct.frac_coords.to('cpu'),
                        coords_are_cartesian=False
                    )
                )

            match_rate = np.array([
                d[0] if (d := matcher.get_rms_dist(s1, s2)) is not None else None for s1, s2 in tqdm(zip(input_list, preds_list))
            ])
            res_log['match_rate'] = np.sum(match_rate != None) / len(match_rate)

            torch.save(model.state_dict(), 'test_ckpt.pt')

    wandb.log(res_log)
