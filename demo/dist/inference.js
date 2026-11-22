'use strict';
ort.env.wasm.wasmPaths = new URL('vendor/', location.href).href;
ort.env.wasm.numThreads = 1;
const modelCache = new Map();
async function getModel(spec,progress){
  if(!modelCache.has(spec.url)){
    const promise=(async()=>{
      progress('Loading models… First use may take a moment.');
      const response=await fetch(spec.url);if(!response.ok)throw Error('Model download failed '+response.status);
      const bytes=await response.arrayBuffer();
      const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');
      if(hash!==spec.sha256)throw Error('Model checksum mismatch');
      const session=await ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
      await session.run({eeg:new ort.Tensor('float32',new Float32Array(3000),[1,1,3000])});return session;
    })();modelCache.set(spec.url,promise);
    promise.catch(()=>modelCache.delete(spec.url));
  }
  return modelCache.get(spec.url);
}
function probabilities(logits){const max=Math.max(...logits),exp=Array.from(logits,v=>Math.exp(v-max)),sum=exp.reduce((a,b)=>a+b,0);return exp.map(v=>v/sum)}
async function inferSample(meta,key,condition,progress=()=>{}){
  const sample=meta.samples.find(s=>s.key===key);if(!sample||!sample.signals[condition])throw Error('Invalid sample or condition');
  const specs=meta.models[sample.fold],baseline=await getModel(specs.baseline,progress),dae=await getModel(specs.dae,progress),augmented=await getModel(specs.augmented,progress);
  progress('Loading EEG sample…');
  const response=await fetch(sample.signals[condition]);if(!response.ok)throw Error('Sample download failed');
  const signals=new Float32Array(await response.arrayBuffer());if(signals.length!==12000)throw Error('Invalid sample size');
  const clean=signals.slice(0,3000),noisy=signals.slice(3000,6000),bp=signals.slice(6000,9000),wave=signals.slice(9000,12000);
  const tensor=v=>new ort.Tensor('float32',v,[1,1,3000]);
  progress('Running E1–E6…');const rows=[];
  async function result(key,title,signal,model,parameter,extra=0){const start=performance.now();const output=await model.run({eeg:tensor(signal)});const probs=probabilities(output.output.data),pred=probs.indexOf(Math.max(...probs));rows.push({key,title,prediction:meta.labels[pred],correct:meta.labels[pred]===sample.truth,probabilities:probs,ms:performance.now()-start+extra,rmse:Math.sqrt(signal.reduce((s,v,i)=>s+(v-clean[i])**2,0)/3000),wave:Array.from(signal),parameter});}
  await result('E1','Clean baseline',clean,baseline,'Original classifier');
  await result('E2','Noisy baseline',noisy,baseline,'Original classifier');
  await result('E3','Band-pass filter',bp,baseline,sample.e3_parameter.replace('4阶','4th order'));
  await result('E4','Wavelet denoising',wave,baseline,sample.e4_parameter.replace('阈值','threshold'));
  const start=performance.now();const reconstructed=await dae.run({eeg:tensor(noisy)});const elapsed=performance.now()-start;
  await result('E5','Denoising autoencoder',new Float32Array(reconstructed.output.data),baseline,'DAE + original classifier',elapsed);
  await result('E6','Noise-augmented model',noisy,augmented,'Noise-trained classifier');
  return {sample:sample.id,fold:sample.fold+1,truth:sample.truth,condition,rows,runtime:'ONNX Runtime Web 1.20.1 / WASM',preprocessing:meta.preprocessing};
}

