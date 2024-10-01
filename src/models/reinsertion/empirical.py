from typing import Optional, List, Union, Dict

import torch
from torch import Tensor
import torch.nn as nn

class EmpiricalSampler(nn.Module):

    def __init__(
            self,
            property_histograms: Union[Tensor, List[Tensor], Dict, List[Dict]],
            node_in_channels: int=None,
            edge_dim: int=None,
            out_properties: int=1,
            globals_dim: Optional[int]=None,
        ):
        super().__init__()

        self.out_properties = out_properties

        if not isinstance(property_histograms, List):
            property_histograms = [property_histograms]

        list_type = type(property_histograms[0])

        if list_type is dict:

            # convert dictionaries to tensors
            hists = []

            for prop in property_histograms:
                hist_list = [(int(k), float(v)) for k, v in prop.items()]
                # tensor of shape (2, n_nonempty_bins)
                hist_nonempty = torch.tensor(hist_list).transpose(0, 1)
                bin_idx, bin_num = hist_nonempty[0].long(), hist_nonempty[1].float()
                # compute max bin for each property
                max_bin = torch.max(bin_idx)
                # create full histogram from zeros
                hist_full = torch.zeros(max_bin + 1)
                # fill in non-empty bins
                hist_full[bin_idx] = bin_num
                hists.append(hist_full)

            property_histograms = hists

        for i, h in enumerate(property_histograms):
            property_histograms[i] = nn.Parameter(h, requires_grad=False)
        
        # e.g. histogram of the number of nodes in the dataset
        self.property_histograms = nn.ParameterList(property_histograms)


    def forward(
            self,
            x: Tensor=None,
            edge_index: Tensor=None,
            edge_attr: Tensor=None,
            batch: Optional[Tensor]=None,
            batch_size: Optional[int]=None,
            y: Optional[Tensor]=None
        ):
        prop_list = []
        
        for prop_hist in self.property_histograms:
            # shape will be (batch_size,)
            prop_list.append(torch.multinomial(prop_hist, num_samples=batch_size, replacement=True))
        
        return torch.cat(prop_list, dim=-1)