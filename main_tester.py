#!/usr/bin/env python
"""
Script corrigido para análise de ataques WEvade
Executa experimentos de repetição de ataques, combinação de ataques diferentes,
e avaliação de performance e qualidade de imagem.
"""

import os
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import argparse
from pathlib import Path
import logging
import re
from concurrent.futures import ThreadPoolExecutor
import itertools

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WEvadeAnalyzer:
    def __init__(self, base_path='.', results_dir='results'):
        self.base_path = Path(base_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Configurações dos experimentos
        self.checkpoints = {
            'standard': './ckpt/coco.pth',
            'adversarial': './ckpt/coco_adv_train.pth'
        }
        
        self.dataset_folder = './dataset/coco/val'
        
        # Tipos de ataques disponíveis
        self.attack_types = {
            'WEvade-W-II': {'script': 'main.py', 'args': {}},
            'WEvade-W-I': {'script': 'main.py', 'args': {'--WEvade-type': 'WEvade-W-I'}},
            'single-tailed': {'script': 'main.py', 'args': {'--detector-type': 'single-tailed'}},
            'binary-search': {'script': 'main.py', 'args': {'--binary-search': 'True'}},
            'post-processing': {'script': 'existing_post_processing.py', 'args': {}},
            'WEvade-B-Q': {'script': 'main_WEvade_B_Q.py', 'args': {'--exp': 'COCO', '--num-attack': '10', '--norm': 'inf'}}
        }
        
        # Configurações para análise de bits
        self.bit_configurations = [16, 30, 64]  # Ajustado para valores mais realistas
        
        # Resultados dos experimentos
        self.results = {
            'repeated_attacks': {},
            'combined_attacks': {},
            'bit_analysis': {}
        }

    def run_command(self, cmd, timeout=3600):
        """Executa comando do sistema com timeout"""
        try:
            logger.info(f"Executando: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"Comando expirou: {' '.join(cmd)}")
            return False, "", "Timeout expired"
        except Exception as e:
            logger.error(f"Erro ao executar comando: {e}")
            return False, "", str(e)

    def encode_watermarks(self):
        """Codifica watermarks nas imagens"""
        logger.info("=== Codificando watermarks ===")
        
        commands = []
        for training_type, checkpoint in self.checkpoints.items():
            exp_name = 'COCO' if training_type == 'standard' else 'COCO-ADV'
            cmd = [
                'python', 'encode_watermarked_images.py',
                '--checkpoint', checkpoint,
                '--dataset-folder', self.dataset_folder,
                '--exp', exp_name
            ]
            commands.append((training_type, cmd))
        
        for training_type, cmd in commands:
            success, stdout, stderr = self.run_command(cmd)
            if not success:
                logger.error(f"Falha na codificação watermark ({training_type}): {stderr}")
            else:
                logger.info(f"Watermark codificado com sucesso ({training_type})")

    def run_single_attack(self, attack_type, training_type, additional_args=None, save_output=True):
        """Executa um único ataque"""
        attack_config = self.attack_types[attack_type]
        checkpoint = self.checkpoints[training_type]
        
        cmd = ['python', attack_config['script'], '--checkpoint', checkpoint, '--dataset-folder', self.dataset_folder]
        
        # Adiciona argumentos específicos do ataque
        if 'args' in attack_config:
            for key, value in attack_config['args'].items():
                cmd.extend([key, value])
        
        # Adiciona argumentos extras se fornecidos
        if additional_args:
            for key, value in additional_args.items():
                cmd.extend([key, str(value)])
        
        # Ajusta experimento para WEvade-B-Q com treinamento adversarial
        if attack_type == 'WEvade-B-Q' and training_type == 'adversarial':
            # Substitui --exp COCO por --exp COCO-ADV
            if '--exp' in cmd:
                idx = cmd.index('--exp') + 1
                cmd[idx] = 'COCO-ADV'
        
        success, stdout, stderr = self.run_command(cmd)
        
        # Salva output completo para análise manual se solicitado
        if save_output and (success or stderr):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            output_filename = f"{attack_type}_{training_type}_{timestamp}.log"
            output_path = self.results_dir / "outputs" / output_filename
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Success: {success}\n")
                f.write("="*50 + " STDOUT " + "="*50 + "\n")
                f.write(stdout)
                f.write("\n" + "="*50 + " STDERR " + "="*50 + "\n")
                f.write(stderr)
            
            logger.info(f"Output salvo em: {output_path}")
        
        return success, stdout, stderr

    def extract_metrics_from_output(self, stdout, stderr):
        """Extrai métricas do output dos comandos com regex mais robustas"""
        metrics = {
            'evasion_rate': 0.0,
            'bit_accuracy': 0.0,
            'perturbation': 0.0,
            'runtime': 0.0,
            'success': False
        }
        
        try:
            # Combina stdout e stderr para busca
            full_output = stdout + "\n" + stderr
            lines = full_output.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Extrai taxa de evasão - padrões comuns nos scripts WEvade
                evasion_patterns = [
                    r'Evasion rate[=\s:]+([0-9]*\.?[0-9]+)',
                    r'Success rate[=\s:]+([0-9]*\.?[0-9]+)',
                    r'Evading rate.*?([0-9]*\.?[0-9]+)'
                ]
                
                for pattern in evasion_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        try:
                            rate = float(match.group(1))
                            # Se valor > 1, assume que está em percentual
                            metrics['evasion_rate'] = rate / 100.0 if rate > 1 else rate
                            metrics['success'] = True
                            break
                        except ValueError:
                            continue
                
                # Extrai bit accuracy
                bit_patterns = [
                    r'Average Bit_acc[=\s:]+([0-9]*\.?[0-9]+)',
                    r'Bit.*?accuracy.*?([0-9]*\.?[0-9]+)',
                    r'bit_acc.*?([0-9]*\.?[0-9]+)'
                ]
                
                for pattern in bit_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        try:
                            metrics['bit_accuracy'] = float(match.group(1))
                            break
                        except ValueError:
                            continue
                
                # Extrai perturbação
                pert_patterns = [
                    r'Average Perturbation[=\s:]+([0-9]*\.?[0-9]+)',
                    r'perturbation.*?([0-9]*\.?[0-9]+)',
                    r'error.*?norm.*?([0-9]*\.?[0-9]+)'
                ]
                
                for pattern in pert_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        try:
                            metrics['perturbation'] = float(match.group(1))
                            break
                        except ValueError:
                            continue
                
                # Extrai tempo de execução
                time_patterns = [
                    r'Time[=\s:]+([0-9]*\.?[0-9]+)',
                    r'runtime.*?([0-9]*\.?[0-9]+)',
                    r'elapsed.*?([0-9]*\.?[0-9]+)'
                ]
                
                for pattern in time_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        try:
                            metrics['runtime'] = float(match.group(1))
                            break
                        except ValueError:
                            continue
            
            # Se não encontrou evasion_rate mas encontrou outros valores, marca como sucesso parcial
            if metrics['bit_accuracy'] > 0 or metrics['perturbation'] > 0:
                metrics['success'] = True
            
            # Log das métricas extraídas para debug
            logger.debug(f"Métricas extraídas: {metrics}")
            
        except Exception as e:
            logger.warning(f"Erro ao extrair métricas: {e}")
            # Em caso de erro, tenta extrair pelo menos alguma informação básica
            if 'success' in stdout.lower() or 'completed' in stdout.lower():
                metrics['success'] = True
        
        return metrics

    def experiment_repeated_attacks(self, max_repetitions=3):
        """Experimenta ataques repetidos"""
        logger.info("=== Experimento: Ataques Repetidos ===")
        
        for attack_type in ['WEvade-W-II', 'WEvade-W-I']:  # Removido WEvade-B-Q por ser mais complexo
            for training_type in ['standard', 'adversarial']:
                logger.info(f"Testando ataques repetidos: {attack_type} com treinamento {training_type}")
                
                attack_results = []
                
                for repetition in range(1, max_repetitions + 1):
                    logger.info(f"Repetição {repetition}/{max_repetitions}")
                    
                    # Para ataques repetidos, modifica parâmetros ligeiramente
                    additional_args = {
                        '--alpha': str(0.1 * repetition),  # Varia learning rate
                        '--epsilon': str(0.01 + 0.005 * (repetition - 1))  # Varia epsilon
                    }
                    
                    success, stdout, stderr = self.run_single_attack(attack_type, training_type, additional_args)
                    
                    if success:
                        metrics = self.extract_metrics_from_output(stdout, stderr)
                        metrics['repetition'] = repetition
                        metrics['alpha'] = additional_args['--alpha']
                        metrics['epsilon'] = additional_args['--epsilon']
                        attack_results.append(metrics)
                        logger.info(f"Repetição {repetition} - Evasão: {metrics['evasion_rate']:.3f}, Bit Acc: {metrics['bit_accuracy']:.3f}")
                    else:
                        logger.error(f"Falha na repetição {repetition}")
                        # Adiciona resultado com falha para manter consistência
                        metrics = self.extract_metrics_from_output(stdout, stderr)
                        metrics['repetition'] = repetition
                        metrics['success'] = False
                        attack_results.append(metrics)
                
                key = f"{attack_type}_{training_type}"
                self.results['repeated_attacks'][key] = attack_results

    def experiment_combined_attacks(self):
        """Experimenta combinações de diferentes ataques"""
        logger.info("=== Experimento: Ataques Combinados ===")
        
        # Define combinações mais simples e realistas
        attack_combinations = [
            ['WEvade-W-I', 'WEvade-W-II'],
            ['WEvade-W-II', 'binary-search'],
            ['post-processing', 'WEvade-W-I']
        ]
        
        for training_type in ['standard', 'adversarial']:
            for combination in attack_combinations:
                logger.info(f"Testando combinação: {' + '.join(combination)} com treinamento {training_type}")
                
                combined_results = {
                    'combination': combination,
                    'individual_results': [],
                    'final_metrics': {}
                }
                
                # Executa cada ataque da combinação sequencialmente
                all_successful = True
                for attack_type in combination:
                    success, stdout, stderr = self.run_single_attack(attack_type, training_type)
                    
                    metrics = self.extract_metrics_from_output(stdout, stderr)
                    metrics['attack_type'] = attack_type
                    metrics['success'] = success
                    combined_results['individual_results'].append(metrics)
                    
                    if not success:
                        logger.error(f"Falha no ataque {attack_type}")
                        all_successful = False
                
                # Calcula métricas finais da combinação
                if combined_results['individual_results']:
                    successful_results = [r for r in combined_results['individual_results'] if r.get('success', False)]
                    
                    if successful_results:
                        final_evasion = np.mean([r['evasion_rate'] for r in successful_results])
                        final_bit_acc = np.mean([r['bit_accuracy'] for r in successful_results])
                        final_pert = np.mean([r['perturbation'] for r in successful_results])
                        total_runtime = sum([r['runtime'] for r in combined_results['individual_results']])
                        
                        combined_results['final_metrics'] = {
                            'evasion_rate': final_evasion,
                            'bit_accuracy': final_bit_acc,
                            'perturbation': final_pert,
                            'runtime': total_runtime,
                            'success': all_successful,
                            'successful_attacks': len(successful_results),
                            'total_attacks': len(combination)
                        }
                    else:
                        combined_results['final_metrics'] = {
                            'evasion_rate': 0.0,
                            'bit_accuracy': 0.0,
                            'perturbation': 0.0,
                            'runtime': sum([r['runtime'] for r in combined_results['individual_results']]),
                            'success': False,
                            'successful_attacks': 0,
                            'total_attacks': len(combination)
                        }
                
                key = f"{'+'.join(combination)}_{training_type}"
                self.results['combined_attacks'][key] = combined_results

    def experiment_bit_configurations(self):
        """Experimenta diferentes configurações de bits"""
        logger.info("=== Experimento: Configurações de Bits ===")
        
        for bits in self.bit_configurations:
            logger.info(f"Testando configuração de {bits} bits")
            
            for attack_type in ['WEvade-W-II']:  # Foca apenas no ataque principal
                for training_type in ['standard', 'adversarial']:
                    
                    # Adiciona parâmetro de bits
                    additional_args = {
                        '--watermark-length': str(bits)
                    }
                    
                    success, stdout, stderr = self.run_single_attack(attack_type, training_type, additional_args)
                    
                    metrics = self.extract_metrics_from_output(stdout, stderr)
                    metrics['bits'] = bits
                    metrics['success'] = success
                    
                    key = f"{attack_type}_{training_type}"
                    if key not in self.results['bit_analysis']:
                        self.results['bit_analysis'][key] = []
                    
                    self.results['bit_analysis'][key].append(metrics)
                    logger.info(f"{bits} bits - Evasão: {metrics['evasion_rate']:.3f}, Bit Acc: {metrics['bit_accuracy']:.3f}")

    def generate_line_plots(self):
        """Gera gráficos de linha para análise"""
        logger.info("=== Gerando Gráficos de Linha ===")
        
        # Gráfico 1: Métrica vs # ataque repetido
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        for key, results in self.results['repeated_attacks'].items():
            if results:
                repetitions = [r['repetition'] for r in results if r.get('success', False)]
                evasion_rates = [r['evasion_rate'] for r in results if r.get('success', False)]
                bit_accuracies = [r['bit_accuracy'] for r in results if r.get('success', False)]
                
                if repetitions:  # Só plota se houver dados válidos
                    ax1.plot(repetitions, evasion_rates, marker='o', label=key)
                    ax2.plot(repetitions, bit_accuracies, marker='s', label=key)
        
        ax1.set_xlabel('Número de Repetições')
        ax1.set_ylabel('Taxa de Evasão')
        ax1.set_title('Taxa de Evasão vs Número de Ataques Repetidos')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_xlabel('Número de Repetições')
        ax2.set_ylabel('Bit Accuracy')
        ax2.set_title('Bit Accuracy vs Número de Ataques Repetidos')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'repeated_attacks_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gráfico 2: Métrica vs Configuração de Bits
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        for key, results in self.results['bit_analysis'].items():
            if results:
                successful_results = [r for r in results if r.get('success', False)]
                if successful_results:
                    bits_config = [r['bits'] for r in successful_results]
                    evasion_rates = [r['evasion_rate'] for r in successful_results]
                    bit_accuracies = [r['bit_accuracy'] for r in successful_results]
                    
                    ax1.plot(bits_config, evasion_rates, marker='o', label=key)
                    ax2.plot(bits_config, bit_accuracies, marker='s', label=key)
        
        ax1.set_xlabel('Configuração de Bits')
        ax1.set_ylabel('Taxa de Evasão')
        ax1.set_title('Taxa de Evasão vs Configuração de Bits')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_xlabel('Configuração de Bits')
        ax2.set_ylabel('Bit Accuracy')
        ax2.set_title('Bit Accuracy vs Configuração de Bits')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'bit_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_tables(self):
        """Gera tabelas com resultados das combinações"""
        logger.info("=== Gerando Tabelas de Resultados ===")
        
        # Tabela de ataques repetidos
        repeated_data = []
        for key, results in self.results['repeated_attacks'].items():
            for result in results:
                if result.get('success', False):
                    attack_type, training = key.split('_', 1)
                    row = {
                        'Ataque': attack_type,
                        'Treinamento': training,
                        'Repetição': result['repetition'],
                        'Taxa Evasão': f"{result['evasion_rate']:.4f}",
                        'Bit Accuracy': f"{result['bit_accuracy']:.4f}",
                        'Perturbação': f"{result['perturbation']:.4f}",
                        'Tempo (s)': f"{result['runtime']:.2f}",
                        'Alpha': result.get('alpha', 'N/A'),
                        'Epsilon': result.get('epsilon', 'N/A')
                    }
                    repeated_data.append(row)
        
        if repeated_data:
            df_repeated = pd.DataFrame(repeated_data)
            df_repeated.to_csv(self.results_dir / 'repeated_attacks_results.csv', index=False)
            logger.info("Tabela de ataques repetidos salva")
        
        # Tabela de ataques combinados
        combined_data = []
        for key, results in self.results['combined_attacks'].items():
            if 'final_metrics' in results and results['final_metrics']:
                training = key.split('_')[-1]
                combination_name = '_'.join(key.split('_')[:-1])
                
                row = {
                    'Combinação': ' + '.join(results['combination']),
                    'Treinamento': training,
                    'Taxa Evasão': f"{results['final_metrics']['evasion_rate']:.4f}",
                    'Bit Accuracy': f"{results['final_metrics']['bit_accuracy']:.4f}",
                    'Perturbação': f"{results['final_metrics']['perturbation']:.4f}",
                    'Tempo Total (s)': f"{results['final_metrics']['runtime']:.2f}",
                    'Ataques Bem-sucedidos': f"{results['final_metrics']['successful_attacks']}/{results['final_metrics']['total_attacks']}",
                    'Sucesso Geral': 'Sim' if results['final_metrics']['success'] else 'Não'
                }
                combined_data.append(row)
        
        if combined_data:
            df_combined = pd.DataFrame(combined_data)
            df_combined.to_csv(self.results_dir / 'combined_attacks_results.csv', index=False)
            logger.info("Tabela de ataques combinados salva")
        
        # Tabela de configurações de bits
        bit_data = []
        for key, results in self.results['bit_analysis'].items():
            for result in results:
                if result.get('success', False):
                    attack_type, training = key.split('_', 1)
                    row = {
                        'Ataque': attack_type,
                        'Treinamento': training,
                        'Bits': result['bits'],
                        'Taxa Evasão': f"{result['evasion_rate']:.4f}",
                        'Bit Accuracy': f"{result['bit_accuracy']:.4f}",
                        'Perturbação': f"{result['perturbation']:.4f}",
                        'Tempo (s)': f"{result['runtime']:.2f}"
                    }
                    bit_data.append(row)
        
        if bit_data:
            df_bits = pd.DataFrame(bit_data)
            df_bits.to_csv(self.results_dir / 'bit_analysis_results.csv', index=False)
            logger.info("Tabela de análise de bits salva")

    def save_results(self):
        """Salva todos os resultados em JSON"""
        logger.info("=== Salvando Resultados Completos ===")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f'complete_results_{timestamp}.json'
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Resultados salvos em: {results_file}")

    def run_complete_analysis(self):
        """Executa análise completa"""
        logger.info("=== Iniciando Análise Completa WEvade ===")
        
        try:
            # Etapa 1: Preparação
            self.encode_watermarks()
            
            # Etapa 2: Experimentos
            self.experiment_repeated_attacks()
            self.experiment_combined_attacks()
            self.experiment_bit_configurations()
            
            # Etapa 3: Visualização e Relatórios
            self.generate_line_plots()
            self.generate_tables()
            self.save_results()
            
            logger.info("=== Análise Completa Finalizada ===")
            
        except Exception as e:
            logger.error(f"Erro durante análise: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='WEvade Comprehensive Attack Analysis')
    parser.add_argument('--base-path', default='.', help='Caminho base do projeto WEvade')
    parser.add_argument('--results-dir', default='results', help='Diretório para salvar resultados')
    parser.add_argument('--max-repetitions', type=int, default=3, help='Número máximo de repetições de ataque')
    
    args = parser.parse_args()
    
    # Verifica se os arquivos necessários existem
    base_path = Path(args.base_path)
    required_files = ['main.py', 'existing_post_processing.py', 'encode_watermarked_images.py']
    
    missing_files = []
    for file in required_files:
        if not (base_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Arquivos necessários não encontrados: {missing_files}")
        return 1
    
    # Executa análise
    analyzer = WEvadeAnalyzer(args.base_path, args.results_dir)
    try:
        analyzer.run_complete_analysis()
        return 0
    except Exception as e:
        logger.error(f"Falha na análise: {e}")
        return 1

if __name__ == "__main__":
    exit(main())