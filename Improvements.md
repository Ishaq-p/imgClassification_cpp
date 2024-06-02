- recursive for padding is slower than the normal version
- conLayer() is slower than the convol2d(), since convol2d is applying the padding itself (not calling on the addPadding())
- applying padding while doing the the convol function at the same time in convol2d() is slower than first adding the padding and then convol.
- the reason why i cannot see the convol2d() in the gprof output is that thr -O3 flag has considered it as an inline function for better optimization
-  