def get_model_tag(config):
    return f'{config.dataset}' \
           f'{"_aprx" if config.aprx else ""}' \
           f'_g{config.gamma}_lr{config.lr}' \
           f'_rf{config.reduce_factor}' \
           f'_rt{config.reduce_tol}' \
           f'_pt{config.tol_P}' \
           f'_zt{config.tol_Z}'
