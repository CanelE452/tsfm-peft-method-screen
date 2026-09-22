from common import *
import torch
from chronos import ChronosBoltPipeline


def main():
    records={}
    for size,pin in read(RESULTS/'UPSTREAM.json')['models'].items():
        pipeline=ChronosBoltPipeline.from_pretrained(pin['path'],device_map='cpu',dtype=torch.bfloat16,local_files_only=True)
        base=pipeline.model
        tied={'encoder_is_shared':base.encoder.embed_tokens is base.shared,
              'decoder_is_shared':base.decoder.embed_tokens is base.shared,
              'encoder_weight_pointer_matches':base.encoder.embed_tokens.weight.data_ptr()==base.shared.weight.data_ptr(),
              'decoder_weight_pointer_matches':base.decoder.embed_tokens.weight.data_ptr()==base.shared.weight.data_ptr()}
        assert all(tied.values())
        records[size]={'tied':tied,'config':base.config.to_dict(),
                       'buffers':{n:{'shape':list(b.shape),'dtype':str(b.dtype)} for n,b in base.named_buffers()},
                       'inspection_device':'cpu','training_updates':0,
                       'scope':'post-fit read-only instance audit; model class/source and persistent state were audited before fits'}
        del base,pipeline
    write(RESULTS/'TIED_CONFIG_RUNTIME_AUDIT.json',records)


if __name__=='__main__':main()
