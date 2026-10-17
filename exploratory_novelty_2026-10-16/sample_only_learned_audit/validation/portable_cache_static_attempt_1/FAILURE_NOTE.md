# Preserved first software-only attempt

The literal publication test failed with Windows WinError5 at `os.link` in this
workspace. No native code/input, actor, environment, audit random stream or
statistical endpoint was loaded. The original wrapper and test bytes are copied
beside this note. The test's temporary synthetic fixture was cleaned by Python's
`TemporaryDirectory`; no scientific files were involved.

The new unfrozen wrapper uses Windows' atomic, non-replacing `os.rename` on
Windows; POSIX retains non-replacing hard-link publication because POSIX rename
can replace a target. This is an engineering correction before any preparation
execution, with no effect on frozen numerical code, inputs or experiment.
