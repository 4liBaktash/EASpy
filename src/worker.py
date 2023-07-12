import numpy as np
from threading import Semaphore
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait, as_completed
import secrets

from .multi_threading_params import _NUM_THREADS

class Worker:

    def __init__(self, n_worker = _NUM_THREADS, mode = "Thread", 
                 seed = secrets.randbelow(1_000_000_000), *args, **kwargs):
        
        self.n_worker = n_worker
        
        if(mode == "Thread"):
            self.executor = ThreadPoolExecutor(self.n_worker, *args, **kwargs)
            
        if(mode == "ThreadRNG"):
            self.executor = ThreadPoolExecutor(self.n_worker, *args, **kwargs)
            seq = np.random.SeedSequence(seed)
            self._random_generators = [np.random.default_rng(s) for s in seq.spawn(self.n_worker)]

            
    
#Multithreadding    
    def run_with_wait(self, function, selector, *args):
        resu = {}
        futures = [self.executor.submit(function, *args, sel) for sel in selector]
        done, not_done = wait(futures, return_when = 'ALL_COMPLETED')
        
        for i,future in enumerate(futures):
            resu[f"{selector[i]}"] = future.result()
            self.catch_exception(future)
            
        self.executor.shutdown(cancel_futures = True)
        return resu
    
    
    def run(self, function, selector, *args):
        resu = {}
        futures = (self.executor.submit(function, *args, sel) for sel in selector)
        
        i = 0
        for future in as_completed(futures):
            resu[f"{i}"] = future.result()
            future.add_done_callback(self.catch_exception)
            i +=1
    
        self.executor.shutdown(cancel_futures = True)
        return resu
    
    
    def catch_exception(self, future):
        if(future.exception()):
            warnings.warn(f"Found exception in worker: {future} !!!")

            
#MultithreadedRNG random.choice()
    @staticmethod
    def _fill_choice(rng_state, out, first, last, _p, _choice):
        out[first:last] = rng_state.choice(_choice, size = (last-first), p = _p)
        
        
    def run_RNG_choice(self, _choice, _size, _p):    
        futures = []
        self.values = np.empty(_size, dtype = np.int32)
        self.step = np.ceil(_size/self.n_worker).astype(np.int_)
        
        for i in range(self.n_worker):
            args = (self._random_generators[i],
                    self.values,
                    i * self.step,
                    (i + 1) * self.step,
                    _p, 
                    _choice)
            
            future = self.executor.submit(Worker._fill_choice, *args)
            futures.append(future)
            
        wait(futures)
        for future in futures:
            self.catch_exception(future)
            
        self.executor.shutdown(cancel_futures = True)   
        return self.values
    
    
#MultithreadedRNG random.random()
    @staticmethod
    def _fill_random(random_state, out, first, last):
        random_state.random(out=out[first:last])
            
    
    def run_RNG_random(self, _size):        
        futures = {}
        self.values = np.empty(_size, dtype = float)
        self.step = np.ceil(_size/self.n_worker).astype(np.int_)
        
        for i in range(self.n_worker):
            args = (Worker._fill_random,
                    self._random_generators[i],
                    self.values,
                    i * self.step,
                    (i + 1) * self.step)
            futures[self.executor.submit(*args)] = i
            
        wait(futures)
        for future in futures:
            self.catch_exception(future)
            
        self.executor.shutdown(cancel_futures = True) 
        return self.values
    
    
#Multiprocessing  
    def run_multiprocessing(self, func, selector, *args):
        "Results are returned in order of selector items"
        resu = {}
        
        with Pool(self.n_worker) as pool:
            TASKS = [(func, *args, sel) for sel in selector]
            results = [pool.apply_async(Worker.proxy_func, t) for t in TASKS]
            
        for i,r in enumerate(results):
            resu[f"{selector[i]}"] = r.get()
                
        return resu
    
    
    @staticmethod
    def proxy_func(func, *args):
        resu = func(*args)
        return resu

    
#in case I forgot it somewhere
    def __del__(self):
        self.executor.shutdown(cancel_futures = True)
