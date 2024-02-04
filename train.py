from dataset import CrystDataset
from torch_geometric.data import DataLoader
from diffusion import CSPDiffusion
import torch
from tqdm import trange, tqdm
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
import wandb

trainset = CrystDataset('train.csv', 'train')
testset = CrystDataset('test.csv', 'test')

train_batch_size = 256
test_batch_size = 128

train_loader = DataLoader(trainset, shuffle=True, batch_size=train_batch_size)
test_loader = DataLoader(testset, shuffle=False, batch_size=test_batch_size)

model = CSPDiffusion()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.6, patience=30, min_lr=1e-4
)

matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

wandb.init(
    project="diff_csp",
    entity='ignat'
)

num_epochs = 1000

for epoch in trange(num_epochs):
    train_loss = []
    res_log = {}
    model.train(True)
    for batch in tqdm(train_loader):
        loss = model.training_step(batch, 0)
        train_loss.append(loss.data.to('cpu').numpy())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    sheduler.step(np.mean(train_loss))
    res_log['train_loss'] = np.mean(train_loss)

    if not epoch % 20:
        model.train(False)
        with torch.no_grad():
            frac_coords, num_atoms, atom_types, lattices, input_data_list = [], [], [], [], []
            i = 0
            for batch in tqdm(test_loader):
                outputs, _ = model.sample(batch)

                frac_coords.append(outputs['frac_coords'].detach().cpu())
                num_atoms.append(outputs['num_atoms'].detach().cpu())
                atom_types.append(outputs['atom_types'].detach().cpu())
                lattices.append(outputs['lattices'].detach().cpu())
                input_data_list = input_data_list + batch.to_data_list()

                i += 1
                if i == 1:
                    break

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
                        species=struct.atom_types,
                        coords=struct.frac_coords,
                        coords_are_cartesian=False
                    )
                )

            match_rate = np.array([matcher.get_rms_dist(s1, s2) for s1, s2 in zip(input_list, preds_list)])
            res_log['match_rate'] = np.sum(match_rate is not None) / len(match_rate)

    wandb.log(res_log)
