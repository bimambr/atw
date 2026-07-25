library(brms)
library(emmeans)
library(bayestestR)
library(performance)
library(ggplot2)
library(loo)

cache_file <- "bayes_models.rds"

df <- read.csv("translations_long.csv")

tqa_table <- aggregate(
  with(df, (3 * accuracy + 2 * acceptability + readability) / 6),
  by = list(treatment = df$treatment),
  FUN = mean
)
cat("TQA (Weighted Average) Summary:\n")
print(tqa_table)

for (col in c("accuracy", "acceptability", "readability")) {
  df[[col]] <- factor(df[[col]], ordered = TRUE, levels = c(1, 2, 3))
}

df$rag_status <- as.factor(df$rag_status)
df$refine_status <- as.factor(df$refine_status)
df$idiom_id <- as.factor(df$idiom_id)

df$rag_status <- relevel(df$rag_status, ref = "RAG-")
df$refine_status <- relevel(df$refine_status, ref = "Refine-")

evaluate_bayes_clmm <- function(response_var, data) {
  cat("\n======================================================\n")
  cat(sprintf(
    "ANALYSING: %s (Bayesian Cumulative Logit)\n",
    toupper(response_var)
  ))
  cat("======================================================\n")

  model_specs <- list(
    PO = list(
      formula = paste(
        response_var,
        "~ rag_status * refine_status + (1 | idiom_id)"
      )
    ),
    NPO = list(
      formula = paste(
        response_var,
        "~ cs(rag_status * refine_status) + (1 | idiom_id)"
      )
    )
  )

  models <- list()

  for (model_name in names(model_specs)) {
    cat("\n---------------------------------\n")
    cat("Fitting", model_name, "model\n")
    cat("---------------------------------\n")
    models[[model_name]] <- brm(
      formula = as.formula(model_specs[[model_name]]$formula),
      data = data,
      family = cumulative(link = "logit"),
      prior = set_prior("normal(0, 1)", class = "b"),
      cores = 4,
      chains = 4,
      iter = 4000,
      warmup = 2000,
      seed = 123,
      control = list(adapt_delta = 0.95),
      save_pars = save_pars(all = TRUE)
    )
  }

  models
}

print_diagnostics <- function(models, metric) {
  for (model_type in names(models)) {
    cat(sprintf(
      "--- %s: %s diagnostics ---\n", toupper(metric), model_type
    ))
    print(describe_posterior(
      models[[model_type]],
      ci = 0.89,
      ci_method = "hdi",
      rope_ci = 1.0
    ))

    cat(sprintf(
      "\n--- %s: %s posterior contrasts ---\n", toupper(metric), model_type
    ))
    em <- emmeans(
      models[[model_type]],
      ~ rag_status * refine_status,
      mode = "latent"
    )
    contrasts <- pairs(em)
    print(describe_posterior(
      contrasts,
      rope_range = rope_range(models[[model_type]]),
      ci = 0.89,
      ci_method = "hdi",
      rope_ci = 1.0
    ))
    cat("\n")
  }

  cat("=====================================\n")
  cat("LOO MODEL COMPARISON\n")
  cat("=====================================\n")
  loo_list <- lapply(models, loo, cores = 4)

  for (nm in names(loo_list)) {
    cat(sprintf("\n--- %s: %s LOO summary ---\n", toupper(metric), nm))
    print(loo_list[[nm]])
  }
  cat("\n")

  pareto_tables <- lapply(loo_list, pareto_k_table)
  comparisons <- loo_compare(loo_list)
  print(comparisons)

  invisible(list(
    loo = loo_list,
    comparison = comparisons,
    pareto_k = pareto_tables
  ))
}

print_ppc_plots <- function(metrics) {
  for (m_type in c("PO", "NPO")) {
    combined_data <- data.frame()

    for (metric_name in names(metrics)) {
      fit <- metrics[[metric_name]][[m_type]]
      y_obs <- as.character(brms::get_y(fit))
      y_rep <- posterior_predict(fit, ndraws = 200)
      categories <- as.character(sort(unique(y_obs)))
      obs_counts <- table(factor(y_obs, levels = categories))

      pred_counts <- apply(y_rep, 1, function(x) {
        table(factor(as.character(x), levels = categories))
      })
      pred_median <- apply(pred_counts, 1, median)
      pred_lower <- apply(pred_counts, 1, quantile, probs = 0.055)
      pred_upper <- apply(pred_counts, 1, quantile, probs = 0.945)

      metric_df <- data.frame(
        metric = toupper(metric_name),
        category = categories,
        obs = as.vector(obs_counts),
        pred_median = as.vector(pred_median),
        pred_lower = as.vector(pred_lower),
        pred_upper = as.vector(pred_upper)
      )

      combined_data <- rbind(combined_data, metric_df)
    }

    combined_data$metric <- factor(
      combined_data$metric,
      levels = c("ACCURACY", "ACCEPTABILITY", "READABILITY")
    )

    # suppress lintr warnings
    category <- metric <- pred_median <- pred_lower <- pred_upper <- obs <- NULL

    p <- ggplot(
      combined_data,
      aes(x = category, group = metric, fill = metric)
    ) +
      geom_col(aes(y = pred_median),
        position = position_dodge(width = 0.8), alpha = 0.8, width = 0.7
      ) +
      geom_errorbar(aes(ymin = pred_lower, ymax = pred_upper),
        position = position_dodge(width = 0.8), width = 0.25, color = "darkgray"
      ) +
      geom_point(aes(y = obs),
        position = position_dodge(width = 0.8),
        shape = 21, fill = "white", color = "black", size = 2
      ) +
      labs(
        x = "Response Category",
        y = "Count",
        fill = "Metric"
      ) +
      theme_minimal() +
      theme(
        legend.position = "bottom",
        panel.grid.minor = element_blank()
      ) +
      scale_fill_brewer(palette = "Set2")

    print(p)
  }
}

print_effect_plots <- function(metrics) {
  for (m_type in c("PO", "NPO")) {
    combined_data <- data.frame()

    for (metric_name in names(metrics)) {
      fit <- metrics[[metric_name]][[m_type]]
      post_desc <- as.data.frame(describe_posterior(
        fit,
        centrality = "median",
        ci = 0.89,
        ci_method = "hdi"
      ))
      post_desc <- post_desc[!grepl("Intercept", post_desc$Parameter), ]
      post_desc$metric <- toupper(metric_name)
      post_desc$Threshold <- ifelse(
        grepl("\\[[0-9]+\\]", post_desc$Parameter),
        paste(
          "Transition", gsub(".*\\[([0-9]+)\\].*", "\\1", post_desc$Parameter)
        ),
        "Constant Effect"
      )
      combined_data <- rbind(combined_data, post_desc)
    }

    combined_data$Parameter <- sapply(combined_data$Parameter, function(x) {
      if (grepl("RAG.*Refine|Refine.*RAG", x, ignore.case = TRUE)) {
        "RAG + Refine"
      } else if (grepl("RAG", x, ignore.case = TRUE)) {
        "RAG"
      } else if (grepl("Refine", x, ignore.case = TRUE)) {
        "Refine"
      } else {
        x
      }
    })
    combined_data$Parameter <- factor(
      combined_data$Parameter,
      levels = rev(c("RAG", "Refine", "RAG + Refine"))
    )
    combined_data$metric <- factor(
      combined_data$metric,
      levels = c("ACCURACY", "ACCEPTABILITY", "READABILITY")
    )

    # nolint start: object_name_linter.
    Parameter <- Median <- CI_low <- CI_high <- metric <- Threshold <- NULL
    # nolint end

    p <- ggplot(combined_data, aes(
      y = Parameter, x = Median, color = metric, shape = Threshold
    )) +
      geom_vline(
        xintercept = 0,
        linetype = "dashed",
        color = "black", alpha = 0.6
      ) +
      geom_pointrange(
        aes(xmin = CI_low, xmax = CI_high),
        position = position_dodge(width = 0.6),
        size = 0.8,
        linewidth = 1
      ) +
      labs(
        x = "Estimate (Log-Odds)",
        y = "Predictor",
        color = "Metric"
      ) +
      theme_minimal() +
      theme(
        legend.position = "bottom",
        legend.box = "vertical",
        legend.spacing.y = unit(0.1, "cm"),
        legend.margin = margin(t = 0, r = 0, b = 0, l = 0),
        panel.grid.minor = element_blank()
      ) +
      scale_color_brewer(palette = "Set2")

    if (m_type == "PO") {
      p <- p + guides(shape = "none")
    }

    print(p)
  }
}

metrics <- if (file.exists(cache_file)) {
  cat("\nUsing cached fitted models from", cache_file, "\n")
  readRDS(cache_file)
} else {
  metrics <- list(
    accuracy      = evaluate_bayes_clmm("accuracy", df),
    acceptability = evaluate_bayes_clmm("acceptability", df),
    readability   = evaluate_bayes_clmm("readability", df)
  )

  saveRDS(metrics, cache_file)
  metrics
}

for (metric in names(metrics)) {
  cat("\n###", toupper(metric), "###\n")
  print_diagnostics(metrics[[metric]], metric)
}

print_ppc_plots(metrics)
print_effect_plots(metrics)
