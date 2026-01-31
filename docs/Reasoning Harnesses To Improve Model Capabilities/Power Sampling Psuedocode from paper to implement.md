function SCALABLE_POWER_SAMPLE(model p, prompt q,
                               alpha,          # power exponent (>=1)
                               T_max,          # max tokens to generate
                               K_schedule,     # K_t per step (Top-K candidates)
                               M_schedule,     # M_t per step (# rollouts per candidate)
                               H_schedule=None # optional lookahead truncation
                              ):

    x = []   # generated tokens so far

    for t in 0 .. T_max-1:

        # 1) Candidate set: restrict to Top-K_t tokens under base model p(.|q,x)
        logits = model_logits(p, q, x)
        G = topK_tokens(logits, K_schedule[t])             # G_t = Top@K_t[p(.|q,x)] :contentReference[oaicite:3]{index=3}

        # 2) For each candidate token a in G, estimate zeta_t(a) using rollouts
        for each token a in G:

            # sample M_t independent continuations conditioned on choosing a now
            rollouts = []
            for r in 1 .. M_schedule[t]:
                x_future = sample_autoregressive(p, q, x + [a],
                                                 max_len = H_schedule[t] if set else (T_max - t - 1))
                rollouts.append(x_future)

            # compute MC estimate of zeta_t(a)
            # (paper: zeta is an expectation over future completions under p; estimated via rollouts) :contentReference[oaicite:4]{index=4}
            zeta_hat[a] = average_over_rollouts( p_prob(rollout | q, x+[a])^(alpha-1) )

            # compute leave-one-out zeta estimates for jackknife
            for s in 1 .. M_schedule[t]:
                zeta_hat_LOO[a, s] = average_over_rollouts_excluding_s(
                                        p_prob(rollout | q, x+[a])^(alpha-1)
                                     )

        # 3) Build jackknife-corrected estimate of the power next-token distribution over G
        # paper: use jackknife to reduce bias; then sample x_t with that distribution :contentReference[oaicite:5]{index=5}
        p_pow_hat_JK = jackknife_power_probs(
                          base_logits = logits restricted to G,
                          alpha = alpha,
                          zeta_hat = zeta_hat,
                          zeta_hat_LOO = zeta_hat_LOO
                       )

        # 4) Sample next token from the jackknife-corrected power distribution
        next_token = sample_from_distribution(p_pow_hat_JK over G)  # :contentReference[oaicite:6]{index=6}
        x.append(next_token)                                        # :contentReference[oaicite:7]{index=7}

        if next_token == EOS: break

    return x
