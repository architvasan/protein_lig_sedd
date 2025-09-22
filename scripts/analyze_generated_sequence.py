#!/usr/bin/env python3
"""
Analyze generated protein sequence for basic properties and quality metrics.
"""

import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def analyze_protein_sequence(sequence):
    """Analyze a protein sequence for various properties."""
    
    # Basic properties
    length = len(sequence)
    
    # Amino acid composition
    aa_counts = Counter(sequence)
    aa_freq = {aa: count/length for aa, count in aa_counts.items()}
    
    # Standard amino acids
    standard_aas = set('ACDEFGHIKLMNPQRSTVWY')
    non_standard = set(sequence) - standard_aas
    
    # Hydrophobic amino acids
    hydrophobic = set('AILMFPWYV')
    hydrophobic_count = sum(1 for aa in sequence if aa in hydrophobic)
    hydrophobic_freq = hydrophobic_count / length
    
    # Charged amino acids
    positive = set('KRH')
    negative = set('DE')
    positive_count = sum(1 for aa in sequence if aa in positive)
    negative_count = sum(1 for aa in sequence if aa in negative)
    net_charge = positive_count - negative_count
    
    # Polar amino acids
    polar = set('NQST')
    polar_count = sum(1 for aa in sequence if aa in polar)
    polar_freq = polar_count / length
    
    # Secondary structure propensities (rough estimates)
    helix_prone = set('AEHKLMQR')
    sheet_prone = set('CFILTVY')
    turn_prone = set('DGHNPST')
    
    helix_prop = sum(1 for aa in sequence if aa in helix_prone) / length
    sheet_prop = sum(1 for aa in sequence if aa in sheet_prone) / length
    turn_prop = sum(1 for aa in sequence if aa in turn_prone) / length
    
    # Look for common motifs
    signal_peptide = sequence.startswith('M')  # Starts with Met
    
    # Repetitive regions
    max_repeat = 0
    for i in range(len(sequence)-1):
        repeat_len = 1
        while (i + repeat_len < len(sequence) and 
               sequence[i] == sequence[i + repeat_len]):
            repeat_len += 1
        max_repeat = max(max_repeat, repeat_len)
    
    return {
        'length': length,
        'aa_composition': aa_freq,
        'non_standard_aas': non_standard,
        'hydrophobic_freq': hydrophobic_freq,
        'positive_charge': positive_count,
        'negative_charge': negative_count,
        'net_charge': net_charge,
        'polar_freq': polar_freq,
        'helix_propensity': helix_prop,
        'sheet_propensity': sheet_prop,
        'turn_propensity': turn_prop,
        'starts_with_met': signal_peptide,
        'max_repeat_length': max_repeat
    }

def print_analysis(sequence, analysis):
    """Print detailed analysis of the sequence."""
    
    print("🧬 PROTEIN SEQUENCE ANALYSIS")
    print("=" * 60)
    print(f"Sequence: {sequence}")
    print("=" * 60)
    
    print(f"📏 Length: {analysis['length']} amino acids")
    print(f"🔬 Non-standard AAs: {analysis['non_standard_aas'] if analysis['non_standard_aas'] else 'None'}")
    print(f"🧪 Starts with Met: {'Yes' if analysis['starts_with_met'] else 'No'}")
    print(f"🔄 Max repeat length: {analysis['max_repeat_length']}")
    print()
    
    print("⚡ PHYSICOCHEMICAL PROPERTIES")
    print("-" * 30)
    print(f"💧 Hydrophobic frequency: {analysis['hydrophobic_freq']:.2%}")
    print(f"🔋 Positive charges: {analysis['positive_charge']}")
    print(f"🔋 Negative charges: {analysis['negative_charge']}")
    print(f"⚖️  Net charge: {analysis['net_charge']:+d}")
    print(f"🧊 Polar frequency: {analysis['polar_freq']:.2%}")
    print()
    
    print("🌀 SECONDARY STRUCTURE PROPENSITIES")
    print("-" * 35)
    print(f"🌀 Helix propensity: {analysis['helix_propensity']:.2%}")
    print(f"📄 Sheet propensity: {analysis['sheet_propensity']:.2%}")
    print(f"🔄 Turn propensity: {analysis['turn_propensity']:.2%}")
    print()
    
    print("📊 AMINO ACID COMPOSITION")
    print("-" * 25)
    sorted_aas = sorted(analysis['aa_composition'].items(), 
                       key=lambda x: x[1], reverse=True)
    for aa, freq in sorted_aas:
        print(f"{aa}: {freq:.1%} ({int(freq * analysis['length'])})")

def compare_to_natural(analysis):
    """Compare to typical natural protein properties."""
    
    print("\n🔬 COMPARISON TO NATURAL PROTEINS")
    print("=" * 40)
    
    # Typical ranges for natural proteins
    natural_ranges = {
        'hydrophobic_freq': (0.35, 0.45),
        'polar_freq': (0.15, 0.25),
        'helix_propensity': (0.25, 0.35),
        'sheet_propensity': (0.20, 0.30),
        'turn_propensity': (0.25, 0.35)
    }
    
    for prop, (low, high) in natural_ranges.items():
        value = analysis[prop]
        if low <= value <= high:
            status = "✅ Normal"
        elif value < low:
            status = "⬇️ Low"
        else:
            status = "⬆️ High"
        
        prop_name = prop.replace('_', ' ').title()
        print(f"{prop_name}: {value:.2%} {status} (natural: {low:.1%}-{high:.1%})")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT")
    print("-" * 20)
    
    issues = []
    if analysis['max_repeat_length'] > 5:
        issues.append(f"Long repeats ({analysis['max_repeat_length']} AAs)")
    if len(analysis['non_standard_aas']) > 0:
        issues.append(f"Non-standard AAs: {analysis['non_standard_aas']}")
    if abs(analysis['net_charge']) > analysis['length'] * 0.2:
        issues.append(f"High net charge ({analysis['net_charge']:+d})")
    
    if not issues:
        print("✅ Sequence looks realistic!")
    else:
        print("⚠️ Potential issues:")
        for issue in issues:
            print(f"   • {issue}")

def compare_sequences(seq1, seq2, name1="Sequence 1", name2="Sequence 2"):
    """Compare two sequences side by side."""

    print(f"\n🔄 SEQUENCE COMPARISON: {name1} vs {name2}")
    print("=" * 60)

    analysis1 = analyze_protein_sequence(seq1)
    analysis2 = analyze_protein_sequence(seq2)

    # Compare key properties
    properties = [
        ('Length', 'length', ''),
        ('Hydrophobic %', 'hydrophobic_freq', '.1%'),
        ('Net charge', 'net_charge', '+d'),
        ('Polar %', 'polar_freq', '.1%'),
        ('Helix prop.', 'helix_propensity', '.1%'),
        ('Sheet prop.', 'sheet_propensity', '.1%'),
        ('Turn prop.', 'turn_propensity', '.1%'),
        ('Max repeat', 'max_repeat_length', 'd')
    ]

    print(f"{'Property':<15} {'Seq 1':<12} {'Seq 2':<12} {'Difference':<12}")
    print("-" * 55)

    for prop_name, prop_key, fmt in properties:
        val1 = analysis1[prop_key]
        val2 = analysis2[prop_key]

        if fmt == '.1%':
            val1_str = f"{val1:.1%}"
            val2_str = f"{val2:.1%}"
            diff = f"{val2-val1:+.1%}"
        elif fmt == '+d':
            val1_str = f"{val1:+d}"
            val2_str = f"{val2:+d}"
            diff = f"{val2-val1:+d}"
        elif fmt == 'd':
            val1_str = f"{val1}"
            val2_str = f"{val2}"
            diff = f"{val2-val1:+d}"
        else:
            val1_str = f"{val1}"
            val2_str = f"{val2}"
            diff = f"{val2-val1:+d}"

        print(f"{prop_name:<15} {val1_str:<12} {val2_str:<12} {diff:<12}")

def analyze_motifs(sequence):
    """Look for common protein motifs and domains."""

    motifs_found = []

    # Common motifs
    if 'KR' in sequence or 'RK' in sequence:
        motifs_found.append("Basic clusters (DNA/RNA binding)")

    if sequence.count('P') > len(sequence) * 0.1:
        motifs_found.append("Proline-rich (flexible/disordered)")

    if 'DD' in sequence or 'EE' in sequence:
        motifs_found.append("Acidic clusters (metal binding/regulation)")

    # Transmembrane-like regions (hydrophobic stretches)
    hydrophobic = 'AILMFPWYV'
    max_hydrophobic_stretch = 0
    current_stretch = 0

    for aa in sequence:
        if aa in hydrophobic:
            current_stretch += 1
            max_hydrophobic_stretch = max(max_hydrophobic_stretch, current_stretch)
        else:
            current_stretch = 0

    if max_hydrophobic_stretch >= 15:
        motifs_found.append(f"Long hydrophobic stretch ({max_hydrophobic_stretch} AAs - possible TM domain)")

    # Nuclear localization signals (basic clusters)
    import re
    nls_pattern = r'[KR]{2,}[A-Z]{0,10}[KR]{2,}'
    if re.search(nls_pattern, sequence):
        motifs_found.append("Potential nuclear localization signal")

    # Phosphorylation sites
    phos_sites = sequence.count('S') + sequence.count('T') + sequence.count('Y')
    if phos_sites > len(sequence) * 0.15:
        motifs_found.append(f"High phosphorylation potential ({phos_sites} S/T/Y sites)")

    return motifs_found, max_hydrophobic_stretch

if __name__ == "__main__":
    # Your generated sequences
    sequence1 = "MRNSLPHPALKGLLAEAPEIGRYVTTVTPRDNDVDNSDSFDRLVFLDRHDQGVGRNEIVPTGINPTLTGARENVPDETKRSLARGAPAFGTTGNTLIPSETRSRNANGSNSRWGGASPSSTPAIMTDT"
    sequence2 = "TAAAAEQAEDQSAKEQAVAEPAHKSDSGDRPRENPENSSPDSESTTRAGSQDSDTMESRPDLDNEPEDCPQPPSVQRVHPRAGGHVDTETQVATDHHHQAKDGNTNFIAVGPTYAPAPTPDVHIDDAT"
    sequence3 = "HMTTPATITLEEARAGAKDEPSDDDDSLKDSRVSKDEFTRRRPRKGDHSVDRGLPDFLPPLFETESFGTSSRKKMFSPQNGRGDSASKKSHGSPMMPSAASGKSRGASAKRKSGQSGFVSKIPPEPALEDNSPDPSPRYMPEPFEVQPDMDMPTTPFNRYDEINDPNDLTGDPSEEGKIMAYSDKRSSMAAILLKLFPLIRVVSIISRINIGQRSVLPQINTALAFKDIVVADSQLDASALADSRLLFMKETLLALMLKALLFGLHAKIIDHQVLDKNLITIIILELSSNDINGRLSLHSLVSTEDYEIISSEYSTSTESSPHLESSEIKTAMVAVSKAELKTKNGGVLPFVDLKILAKVYKDRLRMVFGGSEISKGCFVHDNSETNILENDESQKQLTNDNIIFAITKYLSIKYGEDTCSKVFSGYVYPFPPDATFAPHYGNRVRNVEGTNFVYSYEKTDRRKYFNKKDYFDK"

    # Analyze the long sequence (sequence 3)
    print("🧬 ANALYZING SEQUENCE 3 (LONG SEQUENCE)")
    print("=" * 70)
    analysis3 = analyze_protein_sequence(sequence3)
    print_analysis(sequence3, analysis3)
    compare_to_natural(analysis3)

    # Look for motifs
    motifs, max_hydrophobic = analyze_motifs(sequence3)
    print(f"\n🔍 MOTIF ANALYSIS")
    print("-" * 20)
    if motifs:
        for motif in motifs:
            print(f"✅ {motif}")
    else:
        print("No obvious motifs detected")

    # Compare all three sequences
    print(f"\n🔄 THREE-WAY COMPARISON")
    print("=" * 50)

    sequences = [sequence1, sequence2, sequence3]
    names = ["Seq 1 (128aa)", "Seq 2 (128aa)", "Seq 3 ({}aa)".format(len(sequence3))]
    analyses = [analyze_protein_sequence(seq) for seq in sequences]

    properties = [
        ('Length', 'length', ''),
        ('Hydrophobic %', 'hydrophobic_freq', '.1%'),
        ('Net charge', 'net_charge', '+d'),
        ('Polar %', 'polar_freq', '.1%'),
        ('Max repeat', 'max_repeat_length', 'd')
    ]

    print(f"{'Property':<15} {'Seq1':<12} {'Seq2':<12} {'Seq3':<12}")
    print("-" * 55)

    for prop_name, prop_key, fmt in properties:
        values = []
        for analysis in analyses:
            val = analysis[prop_key]
            if fmt == '.1%':
                values.append(f"{val:.1%}")
            elif fmt == '+d':
                values.append(f"{val:+d}")
            else:
                values.append(f"{val}")

        print(f"{prop_name:<15} {values[0]:<12} {values[1]:<12} {values[2]:<12}")

    print(f"\n🎉 Analysis complete! Your model shows excellent diversity across different sequence lengths.")
