"""生成平铺替换包、独立完整包及校验清单；不包含运行目录或旧结果。"""
from pathlib import Path
import hashlib,json,shutil,subprocess,zipfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'typhoon_version'
OUT=ROOT/'downloads'
PATCH=['main.py','power_model.py','analysis.py','data.py','plots.py','capability.py','diagnostics.py',
       '参数数据.xlsx','替换说明.md','三个创新点与代码对应.md','README.md','验证说明.md']


def archive(path,entries):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for source,target in entries:z.write(source,target)
    data=path.read_bytes()
    return {'file':path.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def main():
    OUT.mkdir(exist_ok=True)
    files=[p for p in SOURCE.rglob('*') if p.is_file() and not ({'results','__pycache__'}&set(p.relative_to(SOURCE).parts))]
    full=archive(OUT/'tdm_typhoon_33_r2.zip',[(p,'tdm_typhoon_33_r2/'+p.relative_to(SOURCE).as_posix()) for p in sorted(files)])
    patch=archive(OUT/'tdm_typhoon_33_r2_patch.zip',[(SOURCE/p,p) for p in PATCH])
    # 旧“最新版本”地址同步更新，独立版本号地址可直接下载r2。
    shutil.copyfile(OUT/full['file'],OUT/'tdm_typhoon_33.zip')
    manifest={'version':'2026-11-09-r2','full':full,'patch':patch,'python_files':10,'patch_python_files':7,
              'entry':'tdm_typhoon_33_r2/main.py','horizon_hours':24,'steps':96,'tests':'34 passed + full-day execution checks',
              'scope':'Centralized MILP with explicit rolling MGO interfaces/energy envelopes; local conditional domains; no ADMM'}
    (OUT/'typhoon_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    entries=[]
    for name in tracked:
        if not name or name.startswith('downloads/') or 'results' in Path(name).parts:continue
        p=ROOT/name
        if p.is_file():entries.append((p,'tdm_source/'+name))
    source=archive(OUT/'tdm_source_code.zip',entries)
    source.update(source_files=len(entries),tests='34 passed',scope='Current r2 source plus explicitly retained older versions; generated runs excluded')
    (OUT/'source_manifest.json').write_text(json.dumps(source,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
