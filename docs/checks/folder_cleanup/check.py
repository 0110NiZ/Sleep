from pathlib import Path
import ast,hashlib,json,subprocess,os,re,sys,importlib.util
R=Path(__file__).resolve().parents[3];os.chdir(R)
meta=json.loads((R/'docs/checks/folder_cleanup/file_map.json').read_text());checks={};errors=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
kept=[]
for name,h in meta['original_sha256'].items():
    if name.startswith(('result/','configs/','AttnSleep/')) or (name.startswith('reports/') and not name.endswith('README.md')) or name.startswith(('patches/','docs/history/')):
        if not (R/name).is_file() or sha(R/name)!=h:errors.append('changed archive: '+name)
        kept.append(name)
checks['unchanged_original_archive_files']=len(kept)
imports=[]
for name,info in meta['imported'].items():
    if Path(name).suffix not in ('.py','.md'):
        if sha(R/name)!=info['sha256']:errors.append('changed import: '+name)
        imports.append(name)
checks['unchanged_imported_figures_and_tables']=len(imports)
links=[]
for base,ds,fs in os.walk(R,followlinks=False):
    ds[:]=[d for d in ds if d not in ('.git','__pycache__')]
    for name in ds+fs:
        p=Path(base)/name
        if p.is_symlink():
            links.append(str(p.relative_to(R)))
            if not p.exists() or not p.resolve().is_relative_to(R):errors.append('bad link: '+str(p))
checks['relative_links_checked']=len(links)
asts=[]
for old,new in meta['moves'].items():
    if old.startswith('scripts/') and new.startswith('code/'):
        source=subprocess.check_output(['git','show',meta['base_commit']+':'+old],text=True)
        changed=(R/new).read_text()
        changed=re.sub(r'\n# Locate shared modules from either the new or legacy command path\.\nimport sys as _layout_sys\nfrom pathlib import Path as _LayoutPath\n_layout_root = .*?\n_layout_sys.path.insert\(0, str\(_layout_root / \'scripts\'\)\)\n_layout_sys.path.insert\(0, str\(_layout_root / \'code\'\)\)\n','\n',changed)
        if len(Path(new).parts)>2:changed=changed.replace('Path(__file__).resolve().parents[2]','Path(__file__).resolve().parents[1]')
        if ast.dump(ast.parse(source))!=ast.dump(ast.parse(changed)):errors.append('unexpected code AST change: '+new)
        asts.append(new)
checks['experiment_code_AST_unchanged_except_import_path_bootstrap']=asts
syntax=[]
for folder in ('code','scripts','visualization'):
    for p in (R/folder).rglob('*.py'):
        ast.parse(p.read_text());syntax.append(str(p.relative_to(R)))
checks['python_syntax_files']=len(syntax)
py=sys.executable
commands=['code/E2_noisy_baseline.py','code/E2_evaluation.py','code/E3_bandpass.py','code/E4_wavelet.py','code/E5_dae.py','code/E6_augmentation.py','code/summarize_results.py','visualization/comparison_plots.py','visualization/clean_performance/plot.py','visualization/reconstruction/plot.py','visualization/method_comparison/plot.py','scripts/run_e2_evaluation_resumable.py','scripts/run_e3_bandpass.py','scripts/run_e4_wavelet.py','scripts/run_e5_dae.py','scripts/run_e6_noise_augmentation.py','scripts/run_e2_evaluation_v2_resumable.py']
out=[]
for cmd in commands:
    if not (R/cmd).exists():
        if 'v2' in cmd:continue
        errors.append('missing command: '+cmd);continue
    p=subprocess.run([py,cmd,'--help'],capture_output=True,text=True,timeout=60)
    out.append({'command':cmd+' --help','returncode':p.returncode})
    if p.returncode:errors.append(cmd+': '+p.stderr[-2000:])
checks['help_only_checks']=out
# Check protocol guard by reading the exact original generator bytes, no EEG/noise/model execution.
manifest=json.loads((R/'result/E5_dae/manifest.json').read_text())['code']
alias=b'\n# Public version-neutral name; protocol remains Noise v2.\napply_noise = apply_noise_v2\n'
b=(R/'scripts/noise_conditions.py').read_bytes()
assert b.endswith(alias)
checks['E6_generator_hash_guard']=hashlib.sha256(b[:-len(alias)]).hexdigest()==manifest['scripts/noise_conditions_v2.py']
if not checks['E6_generator_hash_guard']:errors.append('E6 noise guard mismatch')
for name in ['configs/fivefold_subject_splits.json','configs/noise_conditions_v2.json','AttnSleep/model/model.py']:
    if sha(R/name)!=manifest[name]:errors.append('E6 fixed input hash mismatch: '+name)
checks['E6_fixed_input_hashes_match']=not any('E6' in e for e in errors)
# Check links in current Markdown only; historical documents intentionally retain historical paths.
mds=[R/'README.md',R/'code/README.md',R/'local/README.md',R/'docs/README.md',R/'docs/setup.md',R/'docs/results.md',R/'docs/handover.md',R/'docs/checks/README.md',*[(R/'docs'/ (n+'.md')) for n in ['E1_baseline','E2_noisy_baseline','E3_bandpass','E4_wavelet','E5_dae','E6_augmentation']],*(R/'visualization').glob('**/README.md')]
for p in mds:
    for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
        if '://' in target or target.startswith('#'):continue
        target=target.split('#')[0]
        if target and not (p.parent/target).exists():errors.append('broken current doc link: '+str(p.relative_to(R))+' -> '+target)
checks['current_markdown_files_link_checked']=len(mds)
checks['no_training_inference_or_plot_regeneration']=True
checks['errors']=errors

print(json.dumps(checks,indent=2))
sys.exit(bool(errors))
