"""打包完整R4、23项机制测试、可编辑公式及实际24h数值证据；不分发字体。"""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'resilience_v4'


def main():
    out=ROOT/'downloads';out.mkdir(exist_ok=True)
    files=[p for p in SOURCE.rglob('*') if p.is_file() and not ({'results','__pycache__'}&set(p.relative_to(SOURCE).parts))]
    if any(p.suffix.lower() in ('.ttc','.ttf','.otf') for p in files):raise RuntimeError('不分发字体')
    if any(p.suffix.lower() in ('.png','.svg','.pdf') for p in files):raise RuntimeError('本次缺少原版宋体，不能夹带未经验证的预览')
    entries=[(p,'tdm_resilience_r4/'+p.relative_to(SOURCE).as_posix()) for p in sorted(files)]
    for name in ('logic','plots','regression'):
        entries.append((ROOT/f'tests/test_v4_{name}.py',f'tdm_resilience_r4/tests/test_v4_{name}.py'))
    checks={name:hashlib.sha256(p.read_bytes()).hexdigest() for p,name in entries};path=out/'tdm_resilience_r4.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for source,name in entries:archive.write(source,name)
        archive.writestr('tdm_resilience_r4/文件SHA256.json',json.dumps(checks,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(path) as archive:
        if archive.testzip():raise RuntimeError('压缩包CRC校验失败')
    manifest={'version':'2026-11-10-r4','file':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'files':len(entries)+1,'runtime_python_modules':len(list(SOURCE.glob('*.py'))),'entry':'tdm_resilience_r4/main.py','parameter':'tdm_resilience_r4/参数数据.xlsx','standalone_tests':23,'tests_scope':'R4 mechanisms, saved numerical plot interfaces without rendering, JSON-to-Excel export; see validation report for complete physical rollout audits','plots':'Original SimSun required; no substitute-font previews included','scope':'Central MILP + full-trajectory column continuous coordinator; finite causal crew rules; conditional backward SOC tube; no ADMM/global POMDP proof'}
    (out/'resilience_r4_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
