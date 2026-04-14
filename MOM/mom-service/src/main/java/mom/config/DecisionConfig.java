package mom.config;

import mom.decision.DecisionEngine;
import mom.decision.InMemoryThresholdPolicy;
import mom.decision.ThresholdPolicy;
import mom.decision.Thresholds;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.security.PublicKey;
import java.util.Map;

@Configuration
public class DecisionConfig {

    @Bean
    public ThresholdPolicy thresholdPolicy() {
        // MVP hardcoded; next step: load from DB
        return new InMemoryThresholdPolicy(Map.of(
                "MDF1|A|18mm_standard", new Thresholds(0.68, 0.78),
                "MDF1|Night|18mm_standard", new Thresholds(0.65, 0.75),
                "MDF1|*|*", new Thresholds(0.66, 0.76)
        ));
    }

    @Bean
    public DecisionEngine decisionEngine(ThresholdPolicy thresholdPolicy) {
        return new DecisionEngine(thresholdPolicy);
    }
}
