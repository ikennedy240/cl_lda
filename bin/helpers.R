estimate_effects <- function(data, covariates, model_type = 'glm', focal_covariates = NULL, topics = NULL, transform_y = NULL, grouping_var = NULL, ...){
  if(is.null(topics)){ #if no topic vector gets passed
    # then we grab all of the names that are of the format ^Topic[numbers]$
    topics <- names(data)[str_detect(names(data), '^Topic\\d+$')]
    if(length(topics)==0){
      stop("ERROR: No topics found in Data, please include vector of topics")
    }
  }
  if(model_type == 'glm'){
    right_side <- paste(covariates, collapse = '+')
  }
  if(model_type == 'lmer'){
    require(lme4)
    right_side <- paste(paste(covariates, collapse = '+'), '+ ( 1 |', grouping_var, ')')
  }
  return(map_dfr(topics, topic_coef, 
                 data = data, 
                 right_side = right_side, 
                 model_type = model_type,
                 focal_covariates = focal_covariates,
                 transform_y=transform_y))
}

topic_coef <- function(i, data, right_side, model_type ='glm', focal_covariates = NULL, transform_y = NULL, ...){
  model_type <- get(model_type)
  model_formula <- paste0(transform_y,'(',i, ')~', right_side)
  model_i <- model_type(model_formula,
                        data = data)
  sum_i <- summary(model_i)
  all_covariates <- row.names(sum_i$coefficients)
  row <- tibble(topic = i,
                covariate = all_covariates,
                coefs = sum_i$coefficients[,1], 
                stderrs = sum_i$coefficients[,2]
  )
  if(!is.null(focal_covariates)){
    return(row %>% filter(str_detect(row$covariate, paste0(focal_covariates, collapse = '|'))))
  }
  return(row)
}

logitsimev <- function (x, b, ci = 0.95, constant = 1) 
{
  if (any(class(x) == "counterfactual") && !is.null(x$model)) {
    x <- model.matrix(x$model, x$x)
  }
  else {
    if (any(class(x) == "list")) 
      x <- x$x
    if (is.data.frame(x)) 
      x <- as.matrix(x)
    if (!is.matrix(x)) {
      if (is.matrix(b)) {
        x <- t(x)
        if (!is.na(constant)) {
          x <- append(x, 1, constant - 1)
        }
      }
      else {
        x <- as.matrix(x)
        if (!is.na(constant)) {
          x <- appendmatrix(x, rep(1, nrow(x)), constant)
        }
      }
    }
    else {
      if (!is.na(constant)) {
        x <- appendmatrix(x, rep(1, nrow(x)), constant)
      }
    }
  }
  esims <- nrow(as.matrix(b))
  nscen <- nrow(x)
  nci <- length(ci)
  res <- list(pe = rep(NA, nscen), lower = matrix(NA, nrow = nscen, 
                                                  ncol = nci), upper = matrix(NA, nrow = nscen, ncol = nci))
  for (i in 1:nscen) {
    simmu <- b %*% x[i, ]
    simy <- 1/(1 + exp(-simmu))
    res$pe[i] <- mean(simy)
    for (k in 1:nci) {
      cint <- quantile(simy, probs = c((1 - ci[k])/2, (1 - 
                                                         (1 - ci[k])/2)))
      res$lower[i, k] <- cint[1]
      res$upper[i, k] <- cint[2]
    }
  }
  res$lower <- drop(res$lower)
  res$upper <- drop(res$upper)
  res
}