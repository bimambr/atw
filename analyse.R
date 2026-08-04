library(brms)
library(emmeans)
library(bayestestR)
library(performance)
library(ggplot2)
library(loo)

df <- read.csv("translations_long.csv")

tqa_table <- aggregate(
  cbind(accuracy, acceptability, readability) ~ treatment,
  data = df,
  FUN = function(x) round(mean(x), 2)
)
tqa_table$Weighted_Average <- round((
  3 * tqa_table$accuracy +
    2 * tqa_table$acceptability +
    1 * tqa_table$readability
) / 6, 2)
colnames(tqa_table) <- c(
  "Treatment",
  "Accuracy Mean",
  "Acceptability Mean",
  "Readability Mean",
  "Weighted Average"
)
cat("TQA (Weighted Average) Summary:\n")
print(tqa_table)

for (col in c("accuracy", "acceptability", "readability")) {
  df[[col]] <- factor(df[[col]], ordered = TRUE, levels = c(1, 2, 3))

  cat(sprintf("\n--- %s Frequencies ---\n", toupper(col)))
  freq_table <- table(df$treatment, df[[col]])
  freq_df <- as.data.frame.matrix(freq_table)
  colnames(freq_df) <- paste("Score", colnames(freq_df))
  print(freq_df)
}

df$rag_status <- as.factor(df$rag_status)
df$refine_status <- as.factor(df$refine_status)
df$idiom_id <- as.factor(df$idiom_id)

df$rag_status <- relevel(df$rag_status, ref = "RAG-")
df$refine_status <- relevel(df$refine_status, ref = "Refine-")

fit_bayes_clmm <- function(response_var,
                           data,
                           model_type = "PO",
                           prior = "normal(0, 1)") {
  formula <- switch(model_type,
    PO = paste(
      response_var,
      "~ rag_status * refine_status + (1 | idiom_id)"
    ),
    NPO = paste(
      response_var,
      "~ cs(rag_status * refine_status) + (1 | idiom_id)"
    ),
    stop("model_type must be PO or NPO")
  )

  cat(
    "\nFitting",
    response_var,
    model_type,
    "with prior",
    prior,
    "\n"
  )

  brm(
    formula = as.formula(formula),
    data = data,
    family = cumulative(link = "logit"),
    prior = set_prior(prior, class = "b"),
    cores = 4,
    chains = 4,
    iter = 4000,
    warmup = 2000,
    seed = 123,
    control = list(adapt_delta = 0.95),
    save_pars = save_pars(all = TRUE)
  )
}

print_posterior_summary <- function(obj) {
  post <- describe_posterior(
    obj,
    rope_range = c(-0.18, 0.18),
    ci = 0.89,
    ci_method = "hdi",
    rope_ci = 1.0
  )
  cols_to_exp <- intersect(c("Median", "CI_low", "CI_high"), names(post))
  post[cols_to_exp] <- lapply(post[cols_to_exp], exp)
  print(post)
  invisible(post)
}

print_diagnostics <- function(models, metric) {
  for (model_type in names(models)) {
    cat(sprintf(
      "--- %s: %s random effects diagnostics ---\n",
      toupper(metric), model_type
    ))
    summary <- posterior_summary(
      models[[model_type]],
      probs = c(0.055, 0.945),
    )
    sd_rows <- grep("^sd_", rownames(summary))
    print(round(summary[sd_rows, , drop = FALSE], 2))

    cat(sprintf(
      "\n--- %s: %s fixed effects diagnostics ---\n",
      toupper(metric), model_type
    ))
    print_posterior_summary(models[[model_type]])

    cat(sprintf(
      "\n--- %s: %s posterior contrasts ---\n",
      toupper(metric), model_type
    ))
    em <- emmeans(
      models[[model_type]],
      ~ rag_status * refine_status,
      mode = "latent"
    )
    contrasts <- pairs(em, reverse = TRUE)
    print_posterior_summary(contrasts)
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

    combined_data$Median <- exp(combined_data$Median)
    combined_data$CI_low <- exp(combined_data$CI_low)
    combined_data$CI_high <- exp(combined_data$CI_high)

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
        xintercept = 1,
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
        x = "Odds Ratio (Exponentiated Log-Odds 89% HDI)",
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
      scale_color_brewer(palette = "Set2") +
      scale_x_log10()

    if (m_type == "PO") {
      p <- p + guides(shape = "none")
    }

    print(p)
  }
}

get_arg_value <- function(flag) {
  idx <- match(flag, args)

  if (is.na(idx) || idx == length(args)) {
    return(NULL)
  }

  args[idx + 1]
}

run_prior_sensitivity <- function(targets, data) {
  sensitivity_priors <- c(
    "normal(0,0.5)",
    "normal(0,2)"
  )

  results <- list()

  for (metric in names(targets)) {
    model_type <- targets[[metric]]

    if (is.null(model_type)) {
      next
    }

    if (!model_type %in% c("PO", "NPO")) {
      stop(metric, " must be PO or NPO")
    }

    results[[metric]] <- list()

    for (prior in sensitivity_priors) {
      results[[metric]][[prior]] <-
        fit_bayes_clmm(
          response_var = metric,
          data = data,
          model_type = model_type,
          prior = prior
        )
    }
  }

  results
}

# --- MAIN ANALYSIS ---
cache_file <- "bayes_models.rds"
metrics <- if (file.exists(cache_file)) {
  cat("\nUsing cached fitted models from", cache_file, "\n")
  readRDS(cache_file)
} else {
  metrics <- list()

  for (metric in c(
    "accuracy",
    "acceptability",
    "readability"
  )) {
    metrics[[metric]] <- list(
      PO = fit_bayes_clmm(
        response_var = metric,
        data = df,
        model_type = "PO",
      ),
      NPO = fit_bayes_clmm(
        response_var = metric,
        data = df,
        model_type = "NPO",
      )
    )
  }

  saveRDS(metrics, cache_file)
  metrics
}

for (metric in names(metrics)) {
  cat("\n###", toupper(metric), "###\n")
  print_diagnostics(metrics[[metric]], metric)
}

print_ppc_plots(metrics)
print_effect_plots(metrics)

# --- SENSITIVITY ANALYSIS ---
sensitivity_cache_file <- "bayes_sensitivity_models.rds"
args <- commandArgs(trailingOnly = TRUE)
run_sensitivity <- "--sensitivity" %in% args

selected_models <- list(
  accuracy = get_arg_value("--accuracy"),
  acceptability = get_arg_value("--acceptability"),
  readability = get_arg_value("--readability")
)

for (metric in names(selected_models)) {
  model <- selected_models[[metric]]

  if (!is.null(model) && !model %in% c("PO", "NPO")) {
    stop(
      metric,
      " sensitivity model must be PO or NPO, got: ",
      model
    )
  }
}

if (run_sensitivity) {
  if (all(vapply(selected_models, is.null, logical(1)))) {
    stop(
      "--sensitivity requires at least one model flag ",
      "(e.g., --accuracy PO)"
    )
  }

  sensitivity_models <- if (file.exists(sensitivity_cache_file)) {
    cat(
      "\nUsing cached sensitivity models from",
      sensitivity_cache_file,
      "\n"
    )

    readRDS(sensitivity_cache_file)
  } else {
    sensitivity_models <- run_prior_sensitivity(
      targets = selected_models,
      data = df
    )

    saveRDS(
      sensitivity_models,
      sensitivity_cache_file
    )

    sensitivity_models
  }

  cat("\n===== SENSITIVITY POSTERIOR SUMMARIES =====\n")
  for (metric in names(sensitivity_models)) {
    cat(
      "\n###",
      toupper(metric),
      "###\n"
    )

    for (prior in names(sensitivity_models[[metric]])) {
      cat(
        "\n--- Prior:",
        prior,
        "---\n"
      )

      print_posterior_summary(
        sensitivity_models[[metric]][[prior]]
      )
    }
  }
}
