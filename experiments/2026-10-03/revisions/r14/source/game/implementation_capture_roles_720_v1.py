"""Compose real capture evidence when owned scopes replace old providers.

No old counter is incremented here. Unreplaced sites retain their existing
gate, while each replaced site needs its selected scope's compilation and
capture receipt. These checks run only for newly captured graph entries.
"""
from __future__ import annotations

DICT_ROLES={
    'c64_all_eight':('combo','c64_by_block'),
    'c32_all_seven':('combo','c32_by_block'),
    'c128_all_twelve':('combo','c128_by_block'),
    'c256_all_sixteen':('combo','c256_by_block'),
    'c512_decoder_all_eight':('combo','c512_decoder'),
    'c32_hidden_native_ten':('before_c32_hidden',None),
    'c512_probability_all_sixteen':('before_c512_probability',None),
    'c512_library_all_sixteen':('before_c512_library',None),
    'native_k8_pre_post':('before_native_k8',None),
    'c128_pairwise_twelve':('before_pairwise',None),
    'c64_attention_project_eight':('before_c64_attention',None),
    'c128_attention_project_twelve':('before_c128_attention',None),
    'decoder_merge_native_fma_five':('before_decoder_merge',None),
    'decoder_merge_unround_four':('before_decoder_merge',None),
}
SCALAR_ROLES={
    'post_entry':('before_post_entry',None),
    'post_k8':('combo','post_k8'),
    'post_attention':('combo','post_attention'),
    'post_contiguous_k8':('combo','post_contiguous_k8'),
    'pre_attention_tail':('combo','pre_fused'),
    'post_rgb_history_tail':('before_post_rgb_tail',None),
}

def _calls(value,*,capture=False):
    if value is None:return None
    return dict(getattr(value,'capture_calls',value) if capture else value)

def capture_before(modes):
    from three_structure_combo_v1 import snapshot_calls
    result={'combo':snapshot_calls(modes.combo_calls),'own':{}}
    for key,attr in (
        ('before_c512_library','c512_library_calls'),('before_native_k8','native_k8_calls'),
        ('before_c32_hidden','c32_hidden_calls'),('before_c512_probability','c512_probability_calls'),
        ('before_decoder_merge','decoder_merge_calls'),('before_pairwise','c128_pairwise_calls'),
        ('before_c64_attention','c64_attention_project_calls'),('before_c128_attention','c128_attention_project_calls')):
        result[key]=_calls(getattr(modes,attr,None),capture=key not in ('before_c512_library','before_native_k8'))
    result['before_post_entry']=(modes.post_calls or {}).get('entry')
    result['before_post_rgb_tail']=(modes.post_rgb_tail_calls or {}).get('tail')
    for owner in modes.implementation_calls_720.values():
        for name,child in owner.children.items():
            value=dict(getattr(child,'capture_calls',None) or getattr(child,'calls',{}) or {})
            result['own'][name]=value
            # A whitelisted entry can delegate to its pinned complete scope.
            # Its method uses the defining scope's module name; main still
            # authenticates the receipt against the registered entry below.
            defining=type(child).__module__
            if defining.startswith('audit_') and defining!=name:
                if defining in result['own'] and result['own'][defining]!=value:
                    raise RuntimeError('Two capture providers claimed one defining scope')
                result['own'][defining]=value
    return result

def _value(snapshot,where):
    top,key=where
    value=snapshot.get(top)
    return value.get(key) if key is not None and isinstance(value,dict) else value

def _sites(snapshot,role):
    if role in DICT_ROLES:
        value=_value(snapshot,DICT_ROLES[role])
        if not isinstance(value,dict):
            raise RuntimeError('Replacement role has no actual old provider site universe')
        return {str(key):count for key,count in value.items()}
    if role in SCALAR_ROLES:
        value=_value(snapshot,SCALAR_ROLES[role])
        if type(value) is not int:
            raise RuntimeError('Replacement role has no actual old provider counter')
        return {role:value}
    raise ValueError('Unknown replaced capture role')

def merge_role_proofs(before,after,old_gates,proofs):
    """Pure CPU composition; callers retain the real counters and provenance."""
    result=dict(old_gates);evidence={}
    for module,rows in proofs.items():
        if not isinstance(rows,dict):raise TypeError('Capture replacement evidence must be a role object')
        for role,proof in rows.items():
            if role not in old_gates or not isinstance(proof,dict):
                raise RuntimeError('A scope attempted to replace an absent/unregistered capture gate')
            expected=_sites(before,role);sites=proof.get('replaced_sites')
            new=proof.get('new_sites')
            if (not isinstance(sites,(tuple,list)) or not sites or any(type(s) is not str for s in sites) or
                    len(set(sites))!=len(sites) or not set(sites)<=set(expected) or
                    not isinstance(new,dict) or not new or type(proof.get('passed')) is not bool or
                    any(type(key) is not str or type(delta) is not int or delta <= 0
                        for key,delta in new.items())):
                raise RuntimeError('Capture role has invalid or incomplete replacement proof')
            keys=proof.get('new_counter_keys',{key:key for key in new})
            start=before.get('own',{}).get(module)
            end=after.get('own',{}).get(module)
            if (not isinstance(keys,dict) or set(keys)!=set(new) or
                    any(type(key) is not str for key in keys.values()) or
                    not isinstance(start,dict) or not isinstance(end,dict) or
                    any(key not in start or key not in end for key in keys.values()) or
                    any(type(start[key]) is not int or type(end[key]) is not int
                        for key in keys.values()) or
                    any(end[keys[name]]-start[keys[name]]!=delta for name,delta in new.items())):
                raise RuntimeError('Replacement dispatch proof differs from its actual capture counter')
            row=evidence.setdefault(role,{'replaced':{},'providers':[],'new_paths_passed':True})
            if set(row['replaced']) & set(sites):
                raise RuntimeError('Two scopes claimed the same captured producer/consumer site')
            row['replaced'].update({site:module for site in sites})
            row['providers'].append({'module':module,**proof})
            row['new_paths_passed'] &= proof['passed']
    for role,row in evidence.items():
        old,new=_sites(before,role),_sites(after,role)
        if set(old)!=set(new):raise RuntimeError('Old provider site universe changed during capture')
        remaining={site:new[site]-value for site,value in old.items() if site not in row['replaced']}
        row['remaining_old_sites']=remaining
        exact=(role.endswith(('_ten','_twelve','_eight','_sixteen','_five','_four')) and
               role not in ('c64_all_eight','c128_all_twelve','c256_all_sixteen','c512_library_all_sixteen','c512_decoder_all_eight'))
        row['passed']=row['new_paths_passed'] and all(delta==1 if exact else delta>0 for delta in remaining.values())
        result[role]=row['passed']
    return result,evidence

def capture_after(modes,before,old_gates):
    after=capture_before(modes);proofs={}
    for owner in modes.implementation_calls_720.values():
        for name,child in owner.children.items():
            method=getattr(child,'capture_role_replacements',None)
            if callable(method):proofs[name]=method(before)
    return merge_role_proofs(before,after,old_gates,proofs)


def body_provider_capture_gate(modes, before, c512, k8, *, builds):
    """Verify actual old or replaced C512/K8 providers at a genuine miss.

    A new body executes two warmups and one capture. Replaced providers must
    publish their real capture-counter delta and compiled-resource evidence;
    their bypassed legacy counters are neither required nor incremented.
    """
    if type(builds) is not int or builds < 1:
        raise ValueError('Body provider proof needs an actual positive capture delta')
    roles = {'c512_library_all_sixteen': c512, 'native_k8_pre_post': k8}
    expected = 3 * builds
    gates = {}
    for role, counts in roles.items():
        if not isinstance(counts, dict) or not counts or any(type(v) is not int or v < 0 for v in counts.values()):
            raise RuntimeError('Body provider proof lost its real counter universe')
        gates[role] = all(v == expected for v in counts.values())
    owners = getattr(modes, 'implementation_calls_720', {})
    evidence = {}
    if owners:
        if not isinstance(before, dict):
            raise RuntimeError('Composed body capture requires its original provider snapshot')
        after = capture_before(modes)
        for role, counts in roles.items():
            start, end = _sites(before, role), _sites(after, role)
            if set(start) != set(end) or set(counts) != set(start) or any(
                    end[k] - start[k] != counts[k] for k in start):
                raise RuntimeError('Body provider delta differs from actual owned counters')
        proofs = {}
        for owner in owners.values():
            owner.validate_frame_context()
            for name, child in owner.children.items():
                method = getattr(child, 'capture_role_replacements', None)
                if callable(method):
                    rows = method(before)
                    proofs[name] = {role: row for role, row in rows.items() if role in roles}
        gates, evidence = merge_role_proofs(before, after, gates, proofs)
        for role, row in evidence.items():
            # Preserve the exact original warmup/capture count for every
            # remaining old site, as well as the selected replacement proof.
            row['passed'] = row['passed'] and all(v == expected for v in row['remaining_old_sites'].values())
            gates[role] = row['passed']
    return {'passed': all(gates.values()), 'gates': gates, 'replacement_evidence': evidence,
            'c512_calls': dict(c512), 'k8_calls': dict(k8), 'builds': builds,
            'old_provider_calls_per_site': expected,
            'count_scope': 'two_warmups_and_one_capture_actual_provider_calls'}
