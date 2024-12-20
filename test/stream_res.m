function res = stream_res(A, U, S, V, w)
    [m,n] = size(A);
    i = 1;

    R = size(U,2);
    rnorms_ = zeros(m+n, R);
    
    while (i <= m)
        if (i+w-1 <= m)
            idx = i:i+w-1;
        else
            idx = i:m;
        end
        A_sub = A(idx,:);
        U_sub = U(idx,:);
        V_sub = V(:,:);

        Av = A_sub*V_sub;
        Atu = A_sub'*U_sub;
        for r = 1:R
            s = S(r);
            av = Av;
            atu = Atu;
            sv = s*V_sub(:,r);
            su = s*U_sub(:,r);

            del_av_su = av(:,r)-su;
            del_atu_sv = atu(:,r)-sv;

            rnorms_(m+1:m+n,r) = del_av_su;
            rnorms_(1:m, r) = del_atu_sv;
        end
        i = i + w;
    end
    res = zeros(R,1);
    for i = 1:R
        d = sqrt(norm(rnorms_(:,r)));
        res(i) = d;
    end
end