"""Use Git Credential Manager credentials only in memory for the authorized repository.

No token is written, printed, placed in a URL or passed on a command line.
The only HTTP destination is api.github.com, and redirects are rejected.
"""
import argparse,hashlib,json,os,subprocess,urllib.error,urllib.request
from pathlib import Path

OWNER='gggz114514-oss';NAME='nr-b580';FULL=f'{OWNER}/{NAME}'
OUT=Path('D:/Codex-NR-Experiments/nr-b580/reference/publication-v0.1.0-pre')
PUBLIC=Path(__file__).resolve().parents[2]/'nr-b580-public'
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise RuntimeError('Unexpected redirect refused')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('inspect','create','release','verify'))
    args=p.parse_args()
    env={**os.environ,'GIT_TERMINAL_PROMPT':'0','GCM_INTERACTIVE':'Never'}
    result=subprocess.run(['git','credential','fill'],input=f'protocol=https\nhost=github.com\nusername={OWNER}\n\n',
        text=True,capture_output=True,env=env,cwd=PUBLIC)
    if result.returncode:raise SystemExit('Existing GitHub credential unavailable non-interactively; no credential text printed')
    fields=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    token=fields.get('password')
    if not token:raise SystemExit('Credential response had no usable password/token')
    opener=urllib.request.build_opener(NoRedirect())
    def api(path,method='GET',body=None):
        assert path.startswith('/') and '://' not in path
        data=None if body is None else json.dumps(body,ensure_ascii=False).encode('utf-8')
        request=urllib.request.Request('https://api.github.com'+path,data=data,method=method,headers={
            'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
            'X-GitHub-Api-Version':'2022-11-28','Content-Type':'application/json','User-Agent':'nr-b580-research-publication'})
        try:
            with opener.open(request,timeout=45) as response:return response.status,json.load(response)
        except urllib.error.HTTPError as error:
            if error.code==404:return 404,None
            raise SystemExit(f'GitHub API HTTP {error.code}; response and credentials suppressed')
    code,user=api('/user');assert code==200 and user['login']==OWNER,'Authenticated account differs from authorized owner'
    code,repo=api('/repos/'+FULL)
    if args.action=='create' and code==404:
        code,repo=api('/user/repos','POST',dict(name=NAME,private=False,auto_init=False,
            description='4060-reference-driven reconstruction of a bit-exact NR backend on Intel Arc B580: methods, evidence, source snapshots. Research preview.'))
        assert code==201
    record=dict(action=args.action,authenticated_login=user['login'],repository_exists=code==200 or code==201)
    if repo:
        assert repo['full_name']==FULL
        record.update(repository_url=repo['html_url'],private=repo['private'],default_branch=repo['default_branch'])
    if args.action in ('release','verify'):
        assert repo and not repo['private']
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PUBLIC,text=True).strip()
        code,release=api('/repos/'+FULL+'/releases/tags/v0.1.0-pre')
        if args.action=='release' and code==404:
            code,release=api('/repos/'+FULL+'/releases','POST',dict(tag_name='v0.1.0-pre',target_commitish=commit,
                name='v0.1.0-pre: 4060参考驱动的NR精确恢复研究',
                body=(PUBLIC/'RELEASE_NOTES.md').read_text(encoding='utf-8'),draft=False,prerelease=True,
                generate_release_notes=False))
            assert code==201
        assert release and release['tag_name']=='v0.1.0-pre' and release['prerelease'] and not release['draft']
        code,ref=api('/repos/'+FULL+'/git/ref/tags/v0.1.0-pre');assert code==200
        obj=ref['object']
        if obj['type']=='tag':
            code,obj=api('/repos/'+FULL+'/git/tags/'+obj['sha']);assert code==200;obj=obj['object']
        assert obj['type']=='commit' and obj['sha']==commit
        record.update(release_url=release['html_url'],tag='v0.1.0-pre',prerelease=True,commit=commit)
    OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT/f'github-{args.action}-v1.json'
    if dest.exists():
        suffix=2
        while (OUT/f'github-{args.action}-v{suffix}.json').exists():suffix+=1
        dest=OUT/f'github-{args.action}-v{suffix}.json'
    dest.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
