import os
import h5py
import argparse
from core.models.ffn import FFN
from core.data.utils import *

parser = argparse.ArgumentParser(description='inference script')
parser.add_argument('--data', type=str, default='./data_raw3_focus_250_filter1_top_area64.h5', help='input images')
parser.add_argument('--label', type=str, default='./pred.h5', help='input images')
parser.add_argument('--model', type=str, default='/home/x903102883/FFN_LM/model/ffn.pth', help='path to ffn model')
parser.add_argument('--delta', default=(5, 5, 5), help='delta offset')
parser.add_argument('--input_size', default=(55,55,55), help='input size')
parser.add_argument('--depth', type=int, default=22, help='depth of ffn')
parser.add_argument('--seg_thr', type=float, default=0.5, help='input size')
parser.add_argument('--mov_thr', type=float, default=0.5, help='input size')
parser.add_argument('--act_thr', type=float, default=0.95, help='input size')

args = parser.parse_args()


def run():
    
    model = FFN(in_channels=4, out_channels=1, input_size=args.input_size, delta=args.delta, depth=args.depth).cuda()

    assert os.path.isfile(args.model)

    
    model.load_state_dict(torch.load(args.model))
    model.eval()

    
    with h5py.File(args.data, 'r') as f:
        images = (f['/image'][()].astype(np.float32) - 128) / 33
        # labels = g['label'].value

    
    canva = Canvas(model, images, args.input_size, args.delta, args.seg_thr, args.mov_thr, args.act_thr)
    
    canva.segment_at((159, 121, 158))


    #canva.segment_all()
    
    # result = canva.segmentation
    # 
    # max_value = result.max()
    # indice, count = np.unique(result, return_counts=True)
    # result[result == -1] = 0
    # result = result*(1.0 * 255/max_value)

    """
    rlt_key = []
    rlt_val = []
    result = canva.target_dic
    with h5py.File(args.label, 'w') as g:
        for key, value in result.items():
            rlt_key.append(key)
            rlt_val.append((value > 0).sum())
            g.create_dataset('id_{}'.format(key), data=value.astype(np.uint8), compression='gzip')
    print('label: {}, number: {}'.format(rlt_key, rlt_val))
    """


if __name__ == '__main__':
    run()
