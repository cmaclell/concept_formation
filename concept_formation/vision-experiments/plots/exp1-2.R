library(readr)
library(readr) # Read the csv file
library(ggplot2)  # Basic plot package

exp1 <- read_csv("Documents/GitHub/catastrophic-forgetting-cobweb/experiments/r_plots/Data/exp1.csv")
exp1$TrainSet[exp1$TrainSet == "L0"] <- "S0"
exp1$Model[exp1$Model == "COBWEB/4T"] <- "COBWEB/4V"

exp1$TrainSet <- factor(exp1$TrainSet, levels = c('S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'))
exp1$TestSet <- factor(exp1$TestSet, levels = c('L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'All'))

# Calculate the averages among "rest labels"
df_avg <- exp1 %>% 
  filter(TestSet %in% c("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9")) %>%
  group_by(TrainSet, Model, Label, Seed) %>% 
  summarize(AvgTestAccuracy = mean(TestAccuracy))

# Then remove the "rest labels"
df_filtered <- exp1 %>%
  filter(!TestSet %in% c("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"))

df_rest <- df_avg %>%
  mutate(TestSet = "Rest", TestAccuracy = AvgTestAccuracy) %>%
  select(-AvgTestAccuracy)

df_1 <- rbind(df_rest, df_filtered[,c("TrainSet", "Model", "Label", "Seed", "TestSet", "TestAccuracy")])
df_1$TestSet[df_1$TestSet == "L0"] <- "Chosen"
df_1$TestSet <- factor(df_1$TestSet, levels = c('Chosen', 'Rest', 'All'))

df_chosen_1 = df_1 %>% 
  filter(TestSet == 'Chosen')
df_chosen_1$Model[df_chosen_1$Model == "COBWEB/4V"] <- "COBWEB/4V, 1"
df_chosen_1$Model[df_chosen_1$Model == "fc"] <- "fc, 1"
df_chosen_1$Model[df_chosen_1$Model == "fc-CNN"] <- "fc-CNN, 1"

exp2 <- read_csv("Documents/GitHub/catastrophic-forgetting-cobweb/experiments/r_plots/Data/exp2.csv")

exp2$TrainSet <- factor(exp2$TrainSet, levels = c('S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'))
exp2$TestSet <- factor(exp2$TestSet, levels = c('L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'All'))

# Calculate the averages among "rest labels"
df_avg <- exp2 %>% 
  filter(TestSet %in% c("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9")) %>%
  group_by(TrainSet, Model, Label, Seed) %>% 
  summarize(AvgTestAccuracy = mean(TestAccuracy))

# Then remove the "rest labels"
df_filtered <- exp2 %>%
  filter(!TestSet %in% c("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"))

df_rest <- df_avg %>%
  mutate(TestSet = "Rest", TestAccuracy = AvgTestAccuracy) %>%
  select(-AvgTestAccuracy)

df_2 <- rbind(df_rest, df_filtered[,c("TrainSet", "Model", "Label", "Seed", "TestSet", "TestAccuracy")])
df_2$TestSet[df_2$TestSet == "L0"] <- "Chosen"
df_2$TestSet <- factor(df_2$TestSet, levels = c('Chosen', 'Rest', 'All'))

df_chosen_2 = df_2 %>% 
  filter(TestSet == 'Chosen')
df_chosen_2$Model[df_chosen_2$Model == "COBWEB/4V"] <- "COBWEB/4V, 2"
df_chosen_2$Model[df_chosen_2$Model == "fc"] <- "fc, 2"
df_chosen_2$Model[df_chosen_2$Model == "fc-CNN"] <- "fc-CNN, 2"

df_chosen_1_2 = rbind(df_chosen_1, df_chosen_2)

p_chosen <- ggplot(df_chosen_1_2, aes(x = TrainSet, y = TestAccuracy, color = Model, fill = Model, group = Model)) +
  #geom_line() +
  #geom_ribbon(aes(ymin = TestAccuracy - sd(TestAccuracy), ymax = TestAccuracy + sd(TestAccuracy)), alpha = 0.2) +
  #geom_ribbon(aes(ymin = quantile(TestAccuracy, 0.025), ymax = quantile(TestAccuracy, 0.975)), alpha = 0.2) +
  #geom_errorbar(stat = "summary",
  #fun.data = function(y) data.frame(ymin = median(y) - sd(y), ymax = median(y) + sd(y)),
  #width = 0.2) +
  geom_errorbar(stat = "summary",
                #fun.data = function(y) data.frame(ymin = median(y) - sd(y), ymax = median(y) + sd(y)),
                fun.data = "mean_cl_boot",
                width = 0.2) +
  stat_summary(#fun.data = function(y) data.frame(y = median(y)),
    fun.data = "mean_cl_boot",
    geom = "line",
    size = 1,
    alpha = 0.5) +
  labs(x = 'Incoming Training Split', y = 'Test Accuracy', title = 'Test Accuracy for Chosen Label with More Pre-defined Training Splits, Experiments 1 & 2') +
  theme_minimal() +
  #facet_wrap(~ Model, ncol = 3) +
  scale_color_manual(values = c('COBWEB/4V, 1' = 'red', 'COBWEB/4V, 2' = 'pink', 'fc, 1' = 'darkgreen', 'fc, 2' = 'lightgreen',
                                'fc-CNN, 1' = 'blue', 'fc-CNN, 2' = 'lightblue')) +
  scale_fill_manual(values = c('COBWEB/4V, 1' = 'red', 'COBWEB/4V, 2' = 'pink', 'fc, 1' = 'darkgreen', 'fc, 2' = 'lightgreen',
                               'fc-CNN, 1' = 'blue', 'fc-CNN, 2' = 'lightblue')) +
  theme(plot.title = element_text(hjust = 0.5))  # Center the title
print(p_chosen)
