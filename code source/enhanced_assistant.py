import re
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import statistics

class EnhancedHydraulicAssistant:
    def __init__(self):
        # Enhanced knowledge base with more detailed hydraulic engineering insights
        self.knowledge_base = {
            "hgl_basics": {
                "definition": "Hydraulic Grade Line (HGL) represents the total energy per unit weight of fluid at any point in a pipeline, excluding kinetic energy. It shows the height to which water would rise in a piezometer tube.",
                "components": "HGL = Elevation + Pressure Head",
                "significance": "HGL slopes indicate energy losses due to friction, and sudden drops may indicate equipment or blockages.",
                "normal_patterns": {
                    "gradual_decline": "Expected due to friction losses along pipeline",
                    "pump_increases": "Step increases at PMS1, PMS2, PMS3, PMS4 stations",
                    "river_crossings": "Pressure increases at 80-90km and 165-175km crossings"
                },
                "concerning_patterns": {
                    "sudden_drops": "May indicate blockages, valve issues, or equipment failure",
                    "excessive_slopes": "High friction losses, possible pipe roughness increase",
                    "oscillations": "System instability, possible cavitation or air entrainment",
                    "flat_sections": "Possible stagnation or measurement errors"
                }
            },
            "ocp_specifics": {
                "pipeline_details": {
                    "total_length": "187km from Head to Terminal",
                    "elevation_change": "650m to 60m (590m total drop)",
                    "stations": 7,
                    "pump_stations": 4,
                    "river_crossings": 2
                },
                "operational_ranges": {
                    "flow_rate": "Typical range 2000-4000 m³/h",
                    "density": "Phosphate slurry 1.2-1.6 kg/L",
                    "pressure": "Varies by station, 5-25 bar typical"
                },
                "critical_points": {
                    "PMS1_45km": "First major pump after head station",
                    "VANNE_68km": "Critical valve control point",
                    "River1_80-90km": "First river crossing with elevation gain",
                    "PMS2_101km": "Mid-pipeline pressure boost",
                    "River2_165-175km": "Second river crossing near terminal"
                }
            }
        }
        
        # Conversation context to maintain state
        self.conversation_context = {
            "last_analysis": None,
            "current_model": None,
            "recent_metrics": None,
            "user_focus": None,
            "conversation_history": []
        }
        
        # Pattern matching for more intelligent responses
        self.intent_patterns = {
            "analysis_request": [
                r"analyz[e|ing]", r"interpret", r"explain.*result", r"what.*mean",
                r"curve.*pattern", r"anomal[y|ies]", r"problem", r"issue"
            ],
            "comparison_request": [
                r"compar[e|ing]", r"which.*better", r"best.*model", r"difference",
                r"vs", r"versus", r"against"
            ],
            "optimization_request": [
                r"optim[ize|ization]", r"improv[e|ement]", r"better", r"enhance",
                r"reduce.*loss", r"increase.*efficiency"
            ],
            "troubleshooting": [
                r"error", r"wrong", r"bad", r"poor", r"fail", r"problem",
                r"not.*work", r"issue", r"trouble"
            ],
            "prediction_inquiry": [
                r"predict", r"forecast", r"future", r"next", r"expect",
                r"will.*happen", r"trend"
            ]
        }

    def generate_response(self, user_message: str, context: Dict[str, Any] = None) -> str:
        """Enhanced response generation with context awareness and intelligent analysis"""
        
        # Update conversation context
        self._update_context(user_message, context)
        
        # Detect user intent
        intent = self._detect_intent(user_message)
        
        # Generate contextual response based on intent and available data
        if context and ("predictions" in context or "metrics" in context):
            return self._generate_data_aware_response(user_message, context, intent)
        else:
            return self._generate_knowledge_based_response(user_message, intent)
    
    def _update_context(self, message: str, context: Dict[str, Any] = None):
        """Update conversation context with new information"""
        self.conversation_context["conversation_history"].append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "context": context
        })
        
        # Keep only last 10 messages for memory efficiency
        if len(self.conversation_context["conversation_history"]) > 10:
            self.conversation_context["conversation_history"] = \
                self.conversation_context["conversation_history"][-10:]
        
        if context:
            if "model_name" in context:
                self.conversation_context["current_model"] = context["model_name"]
            if "metrics" in context:
                self.conversation_context["recent_metrics"] = context["metrics"]
            if "predictions" in context:
                self.conversation_context["last_analysis"] = context["predictions"]
    
    def _detect_intent(self, message: str) -> str:
        """Detect user intent using pattern matching"""
        message_lower = message.lower()
        
        for intent, patterns in self.intent_patterns.items():
            if any(re.search(pattern, message_lower) for pattern in patterns):
                return intent
        
        # Check for specific topics
        if any(term in message_lower for term in ['hgl', 'hydraulic', 'pressure', 'head']):
            return "hgl_inquiry"
        elif any(term in message_lower for term in ['lstm', 'gru', 'tcn', 'model']):
            return "model_inquiry"
        elif any(term in message_lower for term in ['station', 'pump', 'pipeline', 'ocp']):
            return "pipeline_inquiry"
        
        return "general"
    
    def _generate_data_aware_response(self, message: str, context: Dict[str, Any], intent: str) -> str:
        """Generate intelligent responses based on actual prediction data"""
        
        if intent == "analysis_request":
            return self._analyze_prediction_results(context)
        elif intent == "comparison_request":
            return self._compare_models(context)
        elif intent == "troubleshooting":
            return self._diagnose_issues(context)
        elif intent == "optimization_request":
            return self._suggest_optimizations(context)
        else:
            return self._provide_contextual_insights(context, message)
    
    def _analyze_prediction_results(self, context: Dict[str, Any]) -> str:
        """Provide detailed analysis of prediction results"""
        
        if "predictions" in context:
            predictions = context["predictions"]
            model_name = context.get("model_name", "Unknown")
            
            # Analyze prediction patterns
            analysis = f"**🔍 HGL Prediction Analysis - {model_name} Model**\n\n"
            
            if isinstance(predictions, dict) and "true_path" in predictions:
                true_path = predictions["true_path"]
                pred_path = predictions["pred_path"]
                
                # Calculate prediction accuracy at key points
                station_analysis = self._analyze_station_predictions(true_path, pred_path)
                analysis += station_analysis + "\n\n"
                
                # Identify concerning patterns
                pattern_analysis = self._identify_hgl_patterns(pred_path)
                analysis += pattern_analysis + "\n\n"
                
                # Operational recommendations
                recommendations = self._generate_operational_recommendations(true_path, pred_path)
                analysis += recommendations
                
            else:
                # Handle multiple model predictions
                analysis += self._analyze_multiple_predictions(predictions)
        
        elif "metrics" in context:
            analysis = self._analyze_model_metrics(context["metrics"])
        
        else:
            analysis = "No prediction data available for analysis. Please run a model first to get detailed insights."
        
        return analysis
    
    def _analyze_station_predictions(self, true_path: List[Dict], pred_path: List[Dict]) -> str:
        """Analyze prediction accuracy at each station"""
        
        # Extract HGL values at station locations (approximate)
        station_distances = [0, 45, 68, 101, 130, 162, 187]  # km
        station_names = ["Head", "PMS1", "VANNE", "PMS2", "PMS3", "PMS4", "Terminal"]
        
        analysis = "**📊 Station-by-Station Analysis:**\n"
        
        for i, (dist, name) in enumerate(zip(station_distances, station_names)):
            # Find closest points in paths
            true_val = self._find_closest_hgl_value(true_path, dist)
            pred_val = self._find_closest_hgl_value(pred_path, dist)
            
            if true_val and pred_val:
                error = abs(true_val - pred_val)
                error_pct = (error / true_val) * 100 if true_val != 0 else 0
                
                status = "✅ Excellent" if error_pct < 2 else "⚠️ Good" if error_pct < 5 else "❌ Needs Attention"
                
                analysis += f"• **{name} ({dist}km):** {status} - Error: {error:.1f}m ({error_pct:.1f}%)\n"
        
        return analysis
    
    def _identify_hgl_patterns(self, pred_path: List[Dict]) -> str:
        """Identify concerning or notable patterns in HGL predictions"""
        
        if not pred_path or len(pred_path) < 3:
            return "**⚠️ Pattern Analysis:** Insufficient data for pattern analysis"
        
        analysis = "**🔍 HGL Pattern Analysis:**\n"
        
        # Calculate slopes between points
        slopes = []
        for i in range(len(pred_path) - 1):
            dx = pred_path[i+1]['x'] - pred_path[i]['x']
            dy = pred_path[i+1]['y'] - pred_path[i]['y']
            if dx != 0:
                slopes.append(dy / dx)
        
        if slopes:
            avg_slope = statistics.mean(slopes)
            max_slope = max(slopes)
            min_slope = min(slopes)
            
            # Analyze slope patterns
            if abs(max_slope) > 10:  # Steep slope threshold
                analysis += f"• **⚠️ Steep Gradient Detected:** Maximum slope {max_slope:.2f} m/km - Check for blockages\n"
            
            if abs(min_slope) > 10:
                analysis += f"• **⚠️ Sharp Drop Detected:** Minimum slope {min_slope:.2f} m/km - Possible equipment issue\n"
            
            # Check for pump station effects
            pump_stations = [45, 101, 130, 162]  # km locations
            for pump_km in pump_stations:
                pump_effect = self._check_pump_effect(pred_path, pump_km)
                if pump_effect:
                    analysis += pump_effect + "\n"
            
            # Overall assessment
            if -2 <= avg_slope <= 0:
                analysis += "• **✅ Normal Hydraulic Behavior:** Average slope within expected range\n"
            else:
                analysis += f"• **⚠️ Unusual Slope Pattern:** Average slope {avg_slope:.2f} m/km - Review system parameters\n"
        
        return analysis
    
    def _check_pump_effect(self, pred_path: List[Dict], pump_km: float) -> str:
        """Check if pump stations show expected pressure increases"""
        
        # Find points before and after pump station
        before_idx = None
        after_idx = None
        
        for i, point in enumerate(pred_path):
            if point['x'] <= pump_km and (before_idx is None or point['x'] > pred_path[before_idx]['x']):
                before_idx = i
            if point['x'] >= pump_km and after_idx is None:
                after_idx = i
                break
        
        if before_idx is not None and after_idx is not None and before_idx != after_idx:
            pressure_increase = pred_path[after_idx]['y'] - pred_path[before_idx]['y']
            
            if pressure_increase > 20:  # Expected pump boost
                return f"• **✅ Pump Station at {pump_km}km:** Good pressure boost (+{pressure_increase:.1f}m)"
            elif pressure_increase > 0:
                return f"• **⚠️ Pump Station at {pump_km}km:** Low pressure boost (+{pressure_increase:.1f}m) - Check pump efficiency"
            else:
                return f"• **❌ Pump Station at {pump_km}km:** No pressure increase detected - Pump may be offline"
        
        return None
    
    def _generate_operational_recommendations(self, true_path: List[Dict], pred_path: List[Dict]) -> str:
        """Generate operational recommendations based on prediction analysis"""
        
        recommendations = "**💡 Operational Recommendations:**\n"
        
        # Calculate overall prediction accuracy
        total_error = 0
        point_count = 0
        
        for true_point, pred_point in zip(true_path, pred_path):
            if abs(true_point['x'] - pred_point['x']) < 1:  # Same location
                total_error += abs(true_point['y'] - pred_point['y'])
                point_count += 1
        
        if point_count > 0:
            avg_error = total_error / point_count
            
            if avg_error < 5:
                recommendations += "• **System Status:** Excellent - Continue current operations\n"
                recommendations += "• **Maintenance:** Standard preventive maintenance schedule\n"
            elif avg_error < 15:
                recommendations += "• **System Status:** Good - Monitor key stations more closely\n"
                recommendations += "• **Action:** Review pump efficiency at stations with higher errors\n"
            else:
                recommendations += "• **System Status:** Attention Required - Investigate high prediction errors\n"
                recommendations += "• **Priority Actions:**\n"
                recommendations += "  - Check pump station performance\n"
                recommendations += "  - Inspect pipeline for blockages\n"
                recommendations += "  - Verify sensor calibration\n"
        
        # Add specific OCP recommendations
        recommendations += "\n**🏭 OCP-Specific Recommendations:**\n"
        recommendations += "• Monitor phosphate slurry density variations\n"
        recommendations += "• Check river crossing pressure differentials\n"
        recommendations += "• Optimize pump scheduling for energy efficiency\n"
        
        return recommendations
    
    def _find_closest_hgl_value(self, path: List[Dict], target_distance: float) -> Optional[float]:
        """Find HGL value at closest distance point"""
        if not path:
            return None
        
        closest_point = min(path, key=lambda p: abs(p['x'] - target_distance))
        return closest_point['y']
    
    def _compare_models(self, context: Dict[str, Any]) -> str:
        """Compare model performance intelligently"""
        
        if "metrics" in context and isinstance(context["metrics"], dict):
            metrics = context["metrics"]
            
            if all(isinstance(v, dict) for v in metrics.values()):
                # Multiple model comparison
                return self._detailed_model_comparison(metrics)
            else:
                # Single model metrics
                return self._single_model_assessment(metrics, context.get("model_name", ""))
        
        return "No model metrics available for comparison. Please run model training first."
    
    def _detailed_model_comparison(self, all_metrics: Dict[str, Dict]) -> str:
        """Provide detailed comparison of multiple models"""
        
        comparison = "**🏆 Comprehensive Model Comparison**\n\n"
        
        # Rank models by R² score
        ranked_models = sorted(all_metrics.items(), key=lambda x: x[1]["R2"], reverse=True)
        
        comparison += "**📈 Performance Ranking:**\n"
        for i, (model, metrics) in enumerate(ranked_models):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            comparison += f"{rank_emoji} **{model}:** R² = {metrics['R2']:.4f}, RMSE = {metrics['RMSE']:.2f}m\n"
        
        # Best model recommendation
        best_model, best_metrics = ranked_models[0]
        comparison += f"\n**🎯 Recommended Model: {best_model}**\n"
        
        if best_metrics["R2"] > 0.95:
            comparison += "• **Excellent performance** - Ready for production use\n"
        elif best_metrics["R2"] > 0.90:
            comparison += "• **Very good performance** - Suitable for operational decisions\n"
        elif best_metrics["R2"] > 0.80:
            comparison += "• **Good performance** - Consider additional tuning\n"
        else:
            comparison += "• **Needs improvement** - Review data quality and features\n"
        
        # Model-specific insights
        comparison += "\n**🔍 Model-Specific Insights:**\n"
        for model, metrics in all_metrics.items():
            if model == "LSTM":
                comparison += f"• **LSTM:** Best for complex temporal patterns (R² = {metrics['R2']:.3f})\n"
            elif model == "GRU":
                comparison += f"• **GRU:** Balanced speed/accuracy (R² = {metrics['R2']:.3f})\n"
            elif model == "TCN":
                comparison += f"• **TCN:** Fastest inference (R² = {metrics['R2']:.3f})\n"
        
        return comparison
    
    def _diagnose_issues(self, context: Dict[str, Any]) -> str:
        """Diagnose potential issues based on results"""
        
        diagnosis = "**🔧 System Diagnosis**\n\n"
        
        if "metrics" in context:
            metrics = context["metrics"]
            
            # Check for poor performance
            if isinstance(metrics, dict):
                if "R2" in metrics:
                    r2 = metrics["R2"]
                    if r2 < 0.7:
                        diagnosis += "**❌ Poor Model Performance Detected:**\n"
                        diagnosis += "• R² score below 0.7 indicates significant prediction errors\n"
                        diagnosis += "• **Possible causes:**\n"
                        diagnosis += "  - Insufficient training data\n"
                        diagnosis += "  - Missing important features\n"
                        diagnosis += "  - Data quality issues\n"
                        diagnosis += "  - Model complexity mismatch\n\n"
                        
                        diagnosis += "**🔧 Recommended Actions:**\n"
                        diagnosis += "• Collect more historical data\n"
                        diagnosis += "• Check sensor calibration\n"
                        diagnosis += "• Verify data preprocessing steps\n"
                        diagnosis += "• Try different model architectures\n"
                    
                if "RMSE" in metrics:
                    rmse = metrics["RMSE"]
                    if rmse > 20:  # High RMSE for HGL predictions
                        diagnosis += "**⚠️ High Prediction Error:**\n"
                        diagnosis += f"• RMSE of {rmse:.1f}m is above acceptable threshold\n"
                        diagnosis += "• This could impact operational decisions\n"
                        diagnosis += "• Consider model retraining with recent data\n"
        
        if "predictions" in context:
            # Analyze prediction patterns for anomalies
            predictions = context["predictions"]
            if isinstance(predictions, dict) and "pred_path" in predictions:
                pred_path = predictions["pred_path"]
                anomalies = self._detect_prediction_anomalies(pred_path)
                if anomalies:
                    diagnosis += "\n**🚨 Prediction Anomalies Detected:**\n"
                    diagnosis += anomalies
        
        if diagnosis == "**🔧 System Diagnosis**\n\n":
            diagnosis += "**✅ System Status: Normal**\n"
            diagnosis += "• No significant issues detected\n"
            diagnosis += "• Model performance within acceptable range\n"
            diagnosis += "• Continue monitoring for any changes\n"
        
        return diagnosis
    
    def _detect_prediction_anomalies(self, pred_path: List[Dict]) -> str:
        """Detect anomalies in prediction patterns"""
        
        if not pred_path or len(pred_path) < 5:
            return ""
        
        anomalies = ""
        
        # Check for unrealistic HGL values
        hgl_values = [point['y'] for point in pred_path]
        min_hgl = min(hgl_values)
        max_hgl = max(hgl_values)
        
        if max_hgl > 800:  # Unrealistically high
            anomalies += f"• **Unrealistic High Pressure:** {max_hgl:.1f}m detected\n"
        
        if min_hgl < 0:  # Negative pressure
            anomalies += f"• **Negative Pressure:** {min_hgl:.1f}m detected - Check model validity\n"
        
        # Check for sudden jumps
        for i in range(len(pred_path) - 1):
            dy = abs(pred_path[i+1]['y'] - pred_path[i]['y'])
            dx = abs(pred_path[i+1]['x'] - pred_path[i]['x'])
            
            if dx > 0 and dy/dx > 50:  # Very steep change
                anomalies += f"• **Sudden Pressure Change:** {dy:.1f}m over {dx:.1f}km at {pred_path[i]['x']:.1f}km\n"
        
        return anomalies
    
    def _suggest_optimizations(self, context: Dict[str, Any]) -> str:
        """Suggest system optimizations based on analysis"""
        
        optimizations = "**⚡ Optimization Suggestions**\n\n"
        
        # Model performance optimizations
        if "metrics" in context:
            metrics = context["metrics"]
            if isinstance(metrics, dict) and "R2" in metrics:
                r2 = metrics["R2"]
                
                optimizations += "**🤖 Model Optimization:**\n"
                if r2 < 0.9:
                    optimizations += "• **Hyperparameter Tuning:** Adjust learning rate, batch size, epochs\n"
                    optimizations += "• **Feature Engineering:** Add derived features (pressure ratios, flow gradients)\n"
                    optimizations += "• **Data Augmentation:** Include more diverse operating conditions\n"
                
                if r2 > 0.95:
                    optimizations += "• **Model Deployment:** Performance excellent - ready for real-time use\n"
                    optimizations += "• **Monitoring Setup:** Implement continuous model performance tracking\n"
        
        # Operational optimizations
        optimizations += "\n**🏭 Operational Optimization:**\n"
        optimizations += "• **Pump Scheduling:** Optimize pump operation based on predicted HGL patterns\n"
        optimizations += "• **Energy Efficiency:** Use predictions to minimize power consumption\n"
        optimizations += "• **Preventive Maintenance:** Schedule maintenance based on predicted stress points\n"
        optimizations += "• **Flow Control:** Adjust flow rates to maintain optimal HGL profiles\n"
        
        # OCP-specific optimizations
        optimizations += "\n**🇲🇦 OCP-Specific Optimizations:**\n"
        optimizations += "• **Slurry Density Control:** Optimize phosphate concentration for transport efficiency\n"
        optimizations += "• **River Crossing Management:** Pre-adjust pressures before crossings\n"
        optimizations += "• **Terminal Delivery:** Ensure consistent pressure at Jadida terminal\n"
        optimizations += "• **Cost Reduction:** Use predictions to reduce energy costs by 10-15%\n"
        
        return optimizations
    
    def _provide_contextual_insights(self, context: Dict[str, Any], message: str) -> str:
        """Provide contextual insights based on available data"""
        
        insights = "**💡 Contextual Insights**\n\n"
        
        # Recent conversation context
        if self.conversation_context["current_model"]:
            insights += f"**Current Focus:** {self.conversation_context['current_model']} model analysis\n\n"
        
        # Data-driven insights
        if "predictions" in context:
            insights += "**📊 Your Current Predictions:**\n"
            insights += "• HGL curve successfully generated across all 7 stations\n"
            insights += "• River crossings and pump effects are modeled\n"
            insights += "• Ready for operational decision making\n\n"
        
        if "metrics" in context:
            insights += "**📈 Performance Summary:**\n"
            metrics = context["metrics"]
            if isinstance(metrics, dict) and "R2" in metrics:
                r2 = metrics["R2"]
                performance = "Excellent" if r2 > 0.9 else "Good" if r2 > 0.8 else "Needs Improvement"
                insights += f"• Model accuracy: {performance} (R² = {r2:.3f})\n"
                insights += f"• Prediction reliability: {'High' if r2 > 0.9 else 'Medium' if r2 > 0.8 else 'Low'}\n\n"
        
        # Actionable recommendations
        insights += "**🎯 Next Steps:**\n"
        insights += "• Ask specific questions about your results\n"
        insights += "• Request detailed station-by-station analysis\n"
        insights += "• Compare different model performances\n"
        insights += "• Get operational recommendations\n"
        
        return insights
    
    def _generate_knowledge_based_response(self, message: str, intent: str) -> str:
        """Generate responses based on knowledge base when no data context is available"""
        
        message_lower = message.lower()
        
        # HGL-related questions
        if intent == "hgl_inquiry" or any(term in message_lower for term in ['hgl', 'hydraulic', 'pressure']):
            return self._handle_hgl_knowledge_questions(message_lower)
        
        # Model-related questions
        elif intent == "model_inquiry" or any(term in message_lower for term in ['lstm', 'gru', 'tcn', 'model']):
            return self._handle_model_knowledge_questions(message_lower)
        
        # Pipeline-related questions
        elif intent == "pipeline_inquiry" or any(term in message_lower for term in ['ocp', 'pipeline', 'station']):
            return self._handle_pipeline_knowledge_questions(message_lower)
        
        # General assistance
        else:
            return self._handle_general_assistance()
    
    def _handle_hgl_knowledge_questions(self, message: str) -> str:
        """Handle HGL-related knowledge questions"""
        
        hgl_info = self.knowledge_base["hgl_basics"]
        
        if "what is" in message or "define" in message:
            return f"""**🔍 Hydraulic Grade Line (HGL) Explained:**

{hgl_info['definition']}

**Formula:** {hgl_info['components']}

**Significance:** {hgl_info['significance']}

**In OCP Pipeline Context:**
The HGL curve shows energy distribution across your 187km phosphate pipeline from Head (650m) to Terminal (60m), accounting for:
• Elevation changes between stations
• Friction losses in pipe segments  
• Energy additions from 4 pump stations
• Pressure variations at river crossings

Upload your data and run predictions to get specific HGL analysis for your system!"""
        
        elif "pattern" in message or "interpret" in message:
            normal = hgl_info["normal_patterns"]
            concerning = hgl_info["concerning_patterns"]
            
            return f"""**📈 HGL Pattern Interpretation Guide:**

**✅ Normal Patterns:**
• **{list(normal.keys())[0].replace('_', ' ').title()}:** {normal['gradual_decline']}
• **{list(normal.keys())[1].replace('_', ' ').title()}:** {normal['pump_increases']}
• **{list(normal.keys())[2].replace('_', ' ').title()}:** {normal['river_crossings']}

**⚠️ Concerning Patterns:**
• **{list(concerning.keys())[0].replace('_', ' ').title()}:** {concerning['sudden_drops']}
• **{list(concerning.keys())[1].replace('_', ' ').title()}:** {concerning['excessive_slopes']}
• **{list(concerning.keys())[2].replace('_', ' ').title()}:** {concerning['oscillations']}

**💡 Pro Tip:** Run your model predictions to get AI-powered analysis of your specific HGL patterns!"""
        
        return f"{hgl_info['definition']}\n\nFor detailed analysis of your specific system, please upload data and run a prediction model."
    
    def _handle_model_knowledge_questions(self, message: str) -> str:
        """Handle model-related knowledge questions"""
        
        if "compare" in message or "difference" in message or "which" in message:
            return """**🤖 AI Model Comparison for HGL Prediction:**

**🧠 LSTM (Long Short-Term Memory):**
• **Strengths:** Excellent for complex temporal dependencies, handles long sequences
• **Best for:** Complex HGL patterns with seasonal variations
• **Speed:** Slower training, moderate inference
• **OCP Use:** Ideal for long-term trend analysis

**⚡ GRU (Gated Recurrent Unit):**
• **Strengths:** Simpler than LSTM, good performance with fewer parameters
• **Best for:** Balanced accuracy and speed requirements
• **Speed:** Faster than LSTM, good inference speed
• **OCP Use:** Great for real-time monitoring applications

**🚀 TCN (Temporal Convolutional Network):**
• **Strengths:** Parallel processing, stable gradients, excellent for long sequences
• **Best for:** Fast inference and training with clear patterns
• **Speed:** Fastest training and inference
• **OCP Use:** Perfect for real-time operational decisions

**💡 Recommendation:** Use the "Train & Compare All" feature to see which performs best on your specific OCP data!"""
        
        elif any(model in message for model in ['lstm', 'gru', 'tcn']):
            model_name = 'LSTM' if 'lstm' in message else 'GRU' if 'gru' in message else 'TCN'
            models = self.knowledge_base["models"]
            model_info = models[model_name]
            
            return f"""**🤖 {model_name} Model for HGL Prediction:**

**Description:** {model_info['description']}

**Key Advantages:** {model_info['advantages']}

**Best Use Case:** {model_info['best_use']}

**For Your OCP Pipeline:**
This model analyzes patterns in your 21 input features (flow, density, pressure at all stations) to predict HGL values across all 7 stations with high accuracy.

**Training Features:** Flow rates, slurry density, and pressure measurements from Head, PMS1, VANNE, PMS2, PMS3, PMS4, and Terminal stations.

Ready to test {model_name} on your data? Upload your CSV and select "{model_name}" for training!"""
        
        return "I can help you understand LSTM, GRU, and TCN models for HGL prediction. What specific aspect would you like to know about?"
    
    def _handle_pipeline_knowledge_questions(self, message: str) -> str:
        """Handle OCP pipeline-related questions"""
        
        ocp_info = self.knowledge_base["ocp_specifics"]
        
        if "station" in message:
            return """**🏭 OCP Pipeline Stations Overview:**

**📍 Station Details:**
• **Head (0km, 650m):** Starting point with initial pumping
• **PMS1 (45km, 680m):** First major pump station, highest elevation
• **VANNE (68km, 515m):** Critical valve control point
• **PMS2 (101km, 360m):** Mid-pipeline pressure boost
• **PMS3 (130km, 110m):** Third pump station, lowest elevation
• **PMS4 (162km, 143m):** Final pump before terminal
• **Terminal (187km, 60m):** End point at El Jadida

**🌊 River Crossings:**
• **Crossing 1:** 80-90km (+100m elevation gain)
• **Crossing 2:** 165-175km (+80m elevation gain)

**⚙️ Operations:** Continuous phosphate slurry transport with automated pump control and pressure monitoring at each station."""
        
        elif "operation" in message or "flow" in message:
            ranges = ocp_info["operational_ranges"]
            return f"""**⚙️ OCP Pipeline Operations:**

**📊 Operational Parameters:**
• **Flow Rate:** {ranges['flow_rate']}
• **Slurry Density:** {ranges['density']}
• **Operating Pressure:** {ranges['pressure']}

**🎯 Critical Control Points:**
• **PMS1 (45km):** First major pump after head station
• **VANNE (68km):** Critical valve control for flow regulation
• **River Crossings:** Pressure management during elevation changes
• **Terminal (187km):** Final delivery pressure control

**🔧 Monitoring Focus:**
• Continuous HGL monitoring across all stations
• Pump efficiency optimization
• Slurry density control for optimal transport
• Energy consumption minimization"""
        
        return f"""**🇲🇦 OCP Phosphate Pipeline System:**

**System Overview:** {ocp_info['pipeline_details']['total_length']} pipeline transporting phosphate slurry from mining operations to El Jadida port.

**Key Statistics:**
• Total elevation drop: {ocp_info['pipeline_details']['elevation_change']}
• Number of stations: {ocp_info['pipeline_details']['stations']}
• Pump stations: {ocp_info['pipeline_details']['pump_stations']}
• River crossings: {ocp_info['pipeline_details']['river_crossings']}

**Your HGL prediction system helps optimize this critical infrastructure for Morocco's phosphate industry!**"""
    
    def _handle_general_assistance(self) -> str:
        """Handle general assistance requests"""
        
        return """**🤖 Enhanced HGL Prediction Assistant**

I'm your intelligent assistant for hydraulic grade line analysis and OCP pipeline optimization!

**🎯 What I Can Help You With:**

**📊 Intelligent Analysis:**
• Deep dive into your HGL prediction results
• Station-by-station performance analysis  
• Pattern recognition and anomaly detection
• Operational recommendations based on your data

**🤖 Model Guidance:**
• LSTM vs GRU vs TCN comparison for your specific data
• Performance optimization suggestions
• Troubleshooting poor predictions
• Feature importance analysis

**🏭 OCP Pipeline Expertise:**
• Station operations and pump efficiency
• River crossing pressure management
• Slurry transport optimization
• Energy cost reduction strategies

**🔍 Smart Diagnostics:**
• Identify concerning HGL patterns
• Predict maintenance needs
• Optimize pump scheduling
• Detect system anomalies

**💡 Quick Examples:**
• "Analyze my latest LSTM results"
• "Why is my model accuracy low?"
• "What does this HGL pattern mean?"
• "How can I optimize pump station performance?"
• "Compare all three models for my data"

**Ready to get started?** Upload your data, run a prediction, and ask me anything about your results!"""

    def _analyze_model_metrics(self, metrics: Dict[str, Any]) -> str:
        """Analyze and interpret model metrics"""
        
        if isinstance(metrics, dict):
            if all(isinstance(v, dict) for v in metrics.values()):
                # Multiple models
                return self._detailed_model_comparison(metrics)
            else:
                # Single model
                return self._single_model_assessment(metrics, "")
        
        return "Unable to analyze metrics. Please ensure model training completed successfully."
    
    def _single_model_assessment(self, metrics: Dict[str, Any], model_name: str) -> str:
        """Assess single model performance"""
        
        if "R2" not in metrics:
            return "Incomplete metrics data. Please run model training again."
        
        r2 = metrics["R2"]
        rmse = metrics.get("RMSE", 0)
        mae = metrics.get("MAE", 0)
        
        assessment = f"**📊 {model_name} Model Performance Assessment**\n\n"
        
        # Overall performance rating
        if r2 > 0.95:
            rating = "🏆 Excellent"
            status = "Production Ready"
        elif r2 > 0.90:
            rating = "🥇 Very Good"  
            status = "Operational Use"
        elif r2 > 0.80:
            rating = "🥈 Good"
            status = "Needs Fine-tuning"
        elif r2 > 0.70:
            rating = "🥉 Fair"
            status = "Requires Improvement"
        else:
            rating = "❌ Poor"
            status = "Major Issues"
        
        assessment += f"**Overall Rating:** {rating}\n"
        assessment += f"**Status:** {status}\n\n"
        
        assessment += f"**📈 Detailed Metrics:**\n"
        assessment += f"• **R² Score:** {r2:.4f} ({(r2*100):.1f}% variance explained)\n"
        assessment += f"• **RMSE:** {rmse:.2f}m (average prediction error)\n"
        assessment += f"• **MAE:** {mae:.2f}m (typical absolute error)\n\n"
        
        # Specific recommendations
        assessment += "**💡 Recommendations:**\n"
        if r2 > 0.95:
            assessment += "• Deploy for real-time HGL monitoring\n"
            assessment += "• Use for operational decision making\n"
            assessment += "• Set up automated alerts for anomalies\n"
        elif r2 > 0.80:
            assessment += "• Consider hyperparameter tuning\n"
            assessment += "• Add more training data if available\n"
            assessment += "• Monitor performance on new data\n"
        else:
            assessment += "• Review data quality and preprocessing\n"
            assessment += "• Check for missing or corrupted features\n"
            assessment += "• Consider different model architecture\n"
            assessment += "• Validate sensor calibration\n"
        
        return assessment
    
    def _analyze_multiple_predictions(self, predictions: Dict[str, Any]) -> str:
        """Analyze predictions from multiple models"""
        
        analysis = "**🔍 Multi-Model Prediction Analysis**\n\n"
        
        model_count = len(predictions)
        analysis += f"**Models Analyzed:** {model_count}\n\n"
        
        # Analyze each model's predictions
        for model_name, pred_data in predictions.items():
            if isinstance(pred_data, dict) and "pred_path" in pred_data:
                pred_path = pred_data["pred_path"]
                
                # Quick pattern analysis for each model
                if pred_path and len(pred_path) > 2:
                    hgl_values = [point['y'] for point in pred_path]
                    min_hgl = min(hgl_values)
                    max_hgl = max(hgl_values)
                    avg_hgl = sum(hgl_values) / len(hgl_values)
                    
                    analysis += f"**{model_name} Predictions:**\n"
                    analysis += f"• HGL Range: {min_hgl:.1f}m - {max_hgl:.1f}m\n"
                    analysis += f"• Average HGL: {avg_hgl:.1f}m\n"
                    
                    # Check for realistic values
                    if min_hgl < 0:
                        analysis += f"• ⚠️ Warning: Negative pressure detected\n"
                    if max_hgl > 800:
                        analysis += f"• ⚠️ Warning: Unrealistically high pressure\n"
                    
                    analysis += "\n"
        
        analysis += "**💡 Recommendation:** Compare model metrics to select the best performer for your operational needs."
        
        return analysis

# Create enhanced assistant instance
enhanced_assistant = EnhancedHydraulicAssistant()
