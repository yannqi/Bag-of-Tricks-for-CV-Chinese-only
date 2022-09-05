import torch
import torch.nn as nn
#! Only Amp
# --------------------------   
# -----Original Version-----
# --------------------------  

# Creates once at the beginning of training
args = {}
args.amp = True
scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

for image, target in data_iter:
    optimizer.zero_grad()
    # Casts operations to mixed precision
    with torch.cuda.amp.autocast(enabled=args.amp):
        output = model(image)
        loss = criterion(output, target)

    # Scales the loss, and calls backward()
    # to create scaled gradients
    scaler.scale(loss).backward()

    # Unscales gradients and calls
    # or skips optimizer.step()
    scaler.step(optimizer)

    # Updates the scale for next iteration
    scaler.update()

#! Amp + clip_grad_norm
# --------------------------   
# -Amp with clip_grad_norm--
# --------------------------   

# Creates once at the beginning of training
args = {}
args.amp = True
scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

for image, target in data_iter:
    optimizer.zero_grad()
    # Casts operations to mixed precision
    with torch.cuda.amp.autocast(enabled=args.amp):
        output = model(image)
        loss = criterion(output, target)

    if scaler is not None:
        scaler.scale(loss).backward()
        if args.clip_grad_norm is not None:
            # we should unscale the gradients of optimizer's assigned params if do gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if args.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()


