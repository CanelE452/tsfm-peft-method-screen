from common import *
from model import *
from engine import Panel
from resource import load_selected
import data


@torch.no_grad()
def main():
    d=data.load();panel=Panel(d['values'],d['sigma'])
    pairs=data.schedule('Q_TRAIN',92201)[0];x=panel.batch(pairs,False)
    records=[]
    for arm in ['Q_FP','Q_STD','Q_LOFTQ','Q_QERA','Q_IO16','Q_FORECAST']:
        for seed in [92201,92202]:
            net,_,_=load_selected(arm,seed)
            original=predict(net,x).cpu()
            with torch.no_grad():
                official=ChronosBoltPipeline(net.get_base_model()).predict(x,prediction_length=64).transpose(1,2).float().cpu()
            assert torch.equal(original,official)
            original_buffers=tensor_hash(net.named_buffers())
            del net;clear()
            net,_,_=load_selected(arm,seed,deployed=True)
            deployed=predict(net,x).cpu()
            assert torch.equal(original,deployed)
            assert tensor_hash(net.named_buffers())==original_buffers
            records.append({'arm':arm,'seed':seed,'serialized_prediction_max_abs':float((deployed-original).abs().max()),
                            'official_native_prediction_exact':True,'buffer_hash':original_buffers,
                            'serialized_buffer_exact':True,'input_shape':list(x.shape)})
            del net;clear()
    write(RESULTS/'ARTIFACT_RELOAD_PARITY.json',{'status':'PASS','records':records,'optimizer_updates':0,
          'note':'post-training parity using saved selected adapters and original versus serialized packed base; no new fitting'})


if __name__=='__main__':main()
