function [S,V,info] = stream_exact( A, nsvals, varargin )

    %% Set algorithm parameters from input or by using defaults
    params = inputParser;
    addParameter(params, 'updatesz',     1, @isscalar);
    parse( params, varargin{:} );

    %% Extract parameters
    updatesz        = params.Results.updatesz;

    %% Inital setup
    [nrow,ncol] = size(A);
    nupdates = ceil( nrow / updatesz );

    %% Store the output for each window
    svalues = zeros( nsvals, nupdates );
    telapse = zeros( nupdates, 1 );

    %% --- Stream A1,...,Af
    irow = 1;
    iter = 0;
    S    = [];
    V    = [];

    while ( irow <= nrow )

        iter = iter+1;
        %% ----- Compute row indices
        if ( irow+updatesz-1 < nrow )
            idx = irow:irow+updatesz-1;
        else
            idx = irow:nrow;
        end

       	%% Update
        fprintf('Solving update %d of %d: rows %d:%d, %d svals\n', ...
            iter, nupdates, idx(1), idx(end), nsvals );

        %% Do update.
        Afull = full(A(idx,:));
        tsvd = tic;
        [~,S,V] = svd( [ S * V'; Afull ], 'econ' );
        telapse(iter) = toc(tsvd);
        S = S(1:nsvals,1:nsvals);
        V = V(:,1:nsvals);

        %% Collect stats
        svalues(:,iter) = diag( S(1:nsvals,1:nsvals) );

        %% Update row index
        irow = irow+updatesz;

    end % end while

    %% Set output
    S = S(1:nsvals,1:nsvals);
    V = V(:,1:nsvals);

    info = struct();
    info.svalues = svalues(1:nsvals,:);
    info.telapse = telapse;
    info.tkernel = tkernel;
end % end stream_kernel_exact
