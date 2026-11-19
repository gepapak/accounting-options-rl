The first own-code pure check terminated at the reflection test because it
incorrectly required every tested decimal settlement/entry pair to admit the
searched exact binary64 antisymmetry. No native source was compiled or called;
no result JSON was written. The floor-map rational and RN-containment checks
had passed before that assertion point. The correction is to count verified
reflections and explicit fail-closed rejections separately, while retaining
exact antisymmetry as mandatory for every admitted reflection. This is a
source-feasibility boundary, not evidence that native coupling has passed.
