"""
Rescue pipeline (Phases 3-5): the two trained models joined into one system.

    frame -> detect (Model 1, optional SAHI) -> track (ByteTrack)
          -> camera-motion compensation -> group people into rescue units
          -> memory (time stranded, water trend) <- severity (Model 2, FloodNet)
          -> risk score in [0, 1] -> Critical / High / Medium / Low
          -> annotated video, priority map, victim table, alerts

Nothing in this package is trained. Entry point: scripts/run_rescue.py.
"""
