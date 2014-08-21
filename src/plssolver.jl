abstract PLSSolver           # type for solving the penalized least squares problem

## Default method.  Some subtypes override these methods

Base.cholfact(s::PLSSolver,RX::Bool=true) = RX ? s.RX : s.L
Base.logdet(s::PLSSolver,RX::Bool=true) = logdet(cholfact(s,RX))

