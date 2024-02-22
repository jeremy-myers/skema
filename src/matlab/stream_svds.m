function [S,V,info] = stream_kernel( A, nsvals, varargin )

    %% Set algorithm parameters from input or by using defaults
    params = inputParser;
    addParameter(params, 'updatesz',     1, @isscalar);
    addParameter(params, 'primme_opts', [], @isstruct);
    parse( params, varargin{:} );

    %% Extract parameters
    updatesz        = params.Results.updatesz;
    primme_opts     = params.Results.primme_opts;

    %% Inital setup
    [nrow,ncol] = size(A);
    nupdates = ceil( nrow / updatesz );

    %% Store the output for each window
    svalues = zeros( nsvals, nupdates );
    nmatvec = zeros( nupdates, 1 );
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
        [~,S,V,~,STATS] = primme_svds( [ S * V'; A(idx,:) ], nsvals, 'L', primme_opts );
        primme_opts.v0 = V(:,1:nsvals); % Update initial guess

        %% Collect stats
        svalues(:,iter) = diag( S(1:nsvals,1:nsvals) );
        nmatvec(iter)   = STATS.numMatvecs;
        telapse(iter)   = STATS.elapsedTime;

        %% Update row index
        irow = irow+updatesz;

    end % end while

    %% Set output
    S = S(1:nsvals,1:nsvals);
    V = V(:,1:nsvals);

    info = struct();
    info.svalues = svalues(1:nsvals,:);
    info.nmatvec = nmatvec;
    info.telapse = telapse;
end % end stream_svds
