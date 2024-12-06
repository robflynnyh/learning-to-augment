from lcasr.utils.augmentation import SpecAugment
import torch

def mask_gen(
        num_masks=[2,4,6,8,10,12],
        mask_widths=[3,5,10,20,30],
        repeats = 128,
        save=True
):
    masks = []
    for num_mask in num_masks:
        for mask_width in mask_widths:
            for _ in range(repeats):
                specaugment = SpecAugment(zero_masking=True, n_time_masks=0, n_freq_masks=num_mask, freq_mask_param=mask_width)
                mask = ~specaugment(torch.ones(1,80,1)).squeeze().to(torch.bool)
                masks.append(mask)
    masks = torch.stack(masks, 0)
    print(masks.shape)
    if save:
        torch.save(masks, 'masks.pt')
    return masks

if __name__ == '__main__':
    mask_gen()