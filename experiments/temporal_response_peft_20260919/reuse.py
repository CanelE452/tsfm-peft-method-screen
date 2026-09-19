"""An adapter checkpoint hash does not identify its frozen B0."""
def base_seed(seed):
    if isinstance(seed,list):
        if len(seed)!=3:return None
        return seed[0]
    return seed if isinstance(seed,int) and not isinstance(seed,bool) else None

def reusable_prediction(prior, selected, panel, kind):
    aliases={'PLAIN':'C2','B0':'C0'}
    same_arm=aliases.get(prior.get('arm'),prior.get('arm'))==aliases.get(selected['arm'],selected['arm'])
    return (prior.get('panel')==panel and prior.get('kind')==kind and same_arm
            and prior.get('source')==selected['source']
            and base_seed(prior.get('seed'))==selected['seed']
            and prior.get('checkpoint_sha256')==selected['sha256'])
