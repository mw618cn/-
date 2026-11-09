"""打包独立R3代码、说明、公式与已验证数值示例；不包含任何字体或替代预览。"""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'resilience_v3'


def main():
    out=ROOT/'downloads';out.mkdir(exist_ok=True)
    files=[p for p in SOURCE.rglob('*') if p.is_file() and not ({'results','__pycache__'}&set(p.relative_to(SOURCE).parts))]
    if any(p.suffix.lower() in ('.ttc','.ttf','.otf') for p in files):raise RuntimeError('禁止无授权分发字体')
    entries=[(p,'tdm_resilience_r3/'+p.relative_to(SOURCE).as_posix()) for p in sorted(files)]
    # 测试脚本在源码目录和独立ZIP中均能定位同一主程序目录。
    entries.append((ROOT/'tests/test_v3_logic.py','tdm_resilience_r3/tests/test_mechanisms.py'))
    checks={name:hashlib.sha256(p.read_bytes()).hexdigest() for p,name in entries}
    path=out/'tdm_resilience_r3.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for source,name in entries:archive.write(source,name)
        archive.writestr('tdm_resilience_r3/文件SHA256.json',json.dumps(checks,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(path) as archive:
        if archive.testzip():raise RuntimeError('压缩包CRC校验失败')
    manifest={'version':'2026-11-10-r3','file':path.name,'bytes':path.stat().st_size,
      'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'files':len(entries)+1,'runtime_python_modules':len(list(SOURCE.glob('*.py'))),
      'entry':'tdm_resilience_r3/main.py','parameter':'tdm_resilience_r3/参数数据.xlsx',
      'tests':'43 repository tests; 9 standalone mechanism tests; full 96-step physical/rolling audit',
      'plots':'Original SimSun required; no substitute-font previews included',
      'scope':'Central MILP/LP and finite causal crew policy class; conditional backward SOC tube; no ADMM/POMDP global proof'}
    (out/'resilience_r3_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
