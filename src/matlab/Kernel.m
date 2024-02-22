classdef (Abstract) Kernel < handle
    properties
        name
    end
    methods (Abstract)
        setArguments(obj,args)
        K = get(obj,X,Y)
    end
end