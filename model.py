from networks import CKMN


def generate_model(opt):

    if opt.model_name == 'CKMN':
        model = CKMN.Event_Model(opt)

        coatt_weight = ['Wb_sc_ob', 'Wv_sc_ob', 'Ws_sc_ob', 'Whv_sc_ob', 'Whs_sc_ob', 'Wb_sc_ac', 
                           'Wv_sc_ac', 'Ws_sc_ac', 'Whv_sc_ac', 'Whs_sc_ac', 'Wb_ob_ac', 'Wv_ob_ac', 
                           'Ws_ob_ac', 'Whv_ob_ac', 'Whs_ob_ac']
        temp_coatt = []
 
        dilated_weight = 'dilated'
        nonlocal_weight = 'nonlocal'
        temp_dnl = []
        
        bn_weight =['scene_dilated_bn1', 'object_dilated_bn1', 'action_dilated_bn1', 'scene_nonlocal_bn1', 'object_nonlocal_bn1', 'action_nonlocal_bn1', 'final_bn1']
        temp_bn = []
     
        scratch_train_module_names = ['dilatednonlocal_bilinear', 'coattention_bilinear', 'final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if k in coatt_weight:
                print('c', k)
                temp_coatt.append(v)

            elif dilated_weight in k:
                print('d', k)
                temp_dnl.append(v)
            elif nonlocal_weight in k:
                print('d', k)
                temp_dnl.append(v)

            elif k[:-5] in bn_weight or k[:-7] in bn_weight:
                print('e', k)
                temp_bn.append(v)

            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('f', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False

        temp = temp_coatt + temp_dnl + temp_bn + temp_scratch
        parameters.append({'params': temp})


    return model, parameters
