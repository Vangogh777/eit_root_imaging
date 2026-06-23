def __getattr__(name):
    if name in {"EITForwardSolver", "EITMeasurement"}:
        from .eit_forward import EITForwardSolver, EITMeasurement
        return {"EITForwardSolver": EITForwardSolver, "EITMeasurement": EITMeasurement}[name]
    if name in {"RootSystemGenerator", "RootSegment"}:
        from .root_simulator import RootSystemGenerator, RootSegment
        return {"RootSystemGenerator": RootSystemGenerator, "RootSegment": RootSegment}[name]
    raise AttributeError(name)
