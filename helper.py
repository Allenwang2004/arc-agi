# Helper if we're reloading the model
import torch, gc, inspect, sys

def clear_old_model_refs():
    """
    Delete `model` and `tokenizer` (if they exist) from the caller
    local *and* global scope, then garbage-collect and free GPU cache.
    """

    # ── figure out the caller’s frame ────────────────────────────
    frm = inspect.currentframe().f_back
    caller_locals  = frm.f_locals
    caller_globals = frm.f_globals

    for var in ("model", "tokenizer"):
        if var in caller_locals:
            try:
                del caller_locals[var]
                if var in sys.modules:   # rarely needed
                    del sys.modules[var]
                print(f"deleted local  {var}")
            except Exception as e:
                print(f"could not delete local {var}: {e}")

        if var in caller_globals:
            try:
                del caller_globals[var]
                print(f"deleted global {var}")
            except Exception as e:
                print(f"could not delete global {var}: {e}")

    # ── Python & CUDA cleanup ───────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU cache cleared.")