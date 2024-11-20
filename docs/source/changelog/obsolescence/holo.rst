[Obsolescence] Remove holography code
=====================================

* The holography implementation is removed. It was partly adapted from HyperSpy
  under GPL v3 and would prevent relicensing of LiberTEM to MIT (:issue:`1649`).
  Since it had only few users, development had moved to
  https://github.com/LiberTEM/LiberTEM-holo, and relicensing to MIT should not
  be delayed, it is removed without deprecation period and will throw an error
  on import explaining the situation. The new holography code will hopefully be
  released in due time. We recommend that users who depend on the previous code
  use an older LiberTEM version (&lt;= 0.14.1) for the time being and migrate to
  the new version when appropriate. (:pr:`1689`)
