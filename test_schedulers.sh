#!/bin/bash

# Test script for mflux-schedulers with Z-Image-Turbo
# Tests all available schedulers with the same prompt and parameters

# Common parameters
MODEL="Tongyi-MAI/Z-Image-Turbo"
WIDTH=832
HEIGHT=1248
STEPS=9
LORA_PATHS=(
    "renderartist/Technically-Color-Z-Image-Turbo"
    "/Users/alessandrorizzo/ai-toolkit/output/my_first_lora_v1test2/my_first_lora_v1test2_000001300.safetensors"
)
LORA_SCALES=(0.5 0.7)

NEGATIVE_PROMPT="blurry, low quality, pixelated, distorted, malformed, noisy, grainy, watermark, text, signature, logo, copyright, realistic skin texture, natural hair color, casual clothing, messy hair, bright lighting, cheerful mood, symmetrical composition, smooth shadows, soft focus, cartoonish, anime, digital painting, watercolor"

PROMPT="High-angle shot looking down at a Sicilian matron lying on blue-tiled cemetery ground, positioned slightly off-center in left third of frame with face in upper portion. Woman in deep black silk organza dress with hand-embroidered velvet lace trim, visible through strategically placed sheer panels revealing gold thread details on upper back. Red hair in elegant updo with loose tendrils framing face, honey-toned skin with porcelain finish, smoky black eyeshadow with winged eyeliner, deep crimson lipstick. Purple aubergine velvet hat with black veil partially covering forehead, hands resting on face in graceful pose with fingers gently touching cheeks, palms open. Extreme foreground shows black umbrella with raindrops creating water reflections on blue tiles, raindrops frozen mid-fall. Midground: woman's body in diagonal pose extending from lower left to upper right, weight distributed through spine with legs straight, left hand on hip, right hand on face. Foreground background: Baroque architecture with gold ornate details and dark blue stone columns, gravestones with intricate carvings. Background: leaden plumbeo sky with rain streaks, crows flying above architecture. Dramatic high-contrast lighting with soft diffused rainlight creating deep shadows on woman's face and dress, warm golden backlight illuminating red hair and lace edges. Black umbrella positioned at 45-degree angle in extreme foreground with visible water droplets reflecting blue tiles, contrasted by deep crimson rose (props) held near left hand. Color palette dominated by cobalt blue (tiles), deep aubergine purple (hat), gold (architecture), black (dress), and deep crimson (rose/hair). Telephoto lens creating slight compression, sharp focus on woman's face and hands, background architecture with motion blur. Textured blue tiles with visible grout lines, matte finish on dress with subtle sheen, smooth skin texture with slight dewiness from rain. Professional photography composition with gestalt geometry, negative space in upper right quadrant, crows positioned along diagonal lines. Rain effects: fine mist and distinct raindrops on umbrella surface, water puddles on tiles reflecting blue sky. Cinematic lighting resembling Helmut Newton's dramatic contrasts, David Lynch's surreal composition, Tarkovsky's atmospheric depth. 8K, ultra HD, high resolution, professional photography, high quality, sharp focus, crisp details, cinematic color grading."

# Output directory
OUTPUT_DIR="scheduler_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "Testing mflux-schedulers with Z-Image-Turbo"
echo "========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Function to run test
run_test() {
    local scheduler_name="$1"
    local scheduler_path="$2"
    local kwargs="$3"
    local output_name="${scheduler_name// /_}"

    echo "-------------------------------------------"
    echo "Testing: $scheduler_name"
    echo "Scheduler: $scheduler_path"
    if [ -n "$kwargs" ]; then
        echo "Parameters: $kwargs"
    fi
    echo "-------------------------------------------"

    if [ -n "$kwargs" ]; then
        mflux-generate-z-image-turbo \
          --model "$MODEL" \
          --width "$WIDTH" \
          --height "$HEIGHT" \
          --seed 1010 \
          --steps "$STEPS" \
          --lora-paths "${LORA_PATHS[@]}" \
          --lora-scales "${LORA_SCALES[@]}" \
          --scheduler "$scheduler_path" \
          --scheduler-kwargs "$kwargs" \
          --negative-prompt "$NEGATIVE_PROMPT" \
          --prompt "$PROMPT" \
          --output "$OUTPUT_DIR/${output_name}.png" \
          --metadata
    else
        mflux-generate-z-image-turbo \
          --model "$MODEL" \
          --width "$WIDTH" \
          --height "$HEIGHT" \
          --seed 1010 \
          --steps "$STEPS" \
          --lora-paths "${LORA_PATHS[@]}" \
          --lora-scales "${LORA_SCALES[@]}" \
          --scheduler "$scheduler_path" \
          --negative-prompt "$NEGATIVE_PROMPT" \
          --prompt "$PROMPT" \
          --output "$OUTPUT_DIR/${output_name}.png" \
          --metadata
    fi

    if [ $? -eq 0 ]; then
        echo "✅ Success: $scheduler_name"
    else
        echo "❌ Failed: $scheduler_name"
    fi
    echo ""
}

# Note: After latest mflux updates, Linear = DDIM Flow = STORK (all use same base Euler method)
# The real differences are in:
# - DDIM: Quadratic timestep spacing
# - STORK: Higher-order Runge-Kutta with Taylor approximations
# - Advanced schedulers: Different noise schedules

# Test 1: DDIM with stochasticity (eta > 0 makes it truly different)
run_test "DDIM Stochastic" "mflux.contrib.schedulers.ddim" '{"eta": 0.3}'

# Test 2: STORK-2 (Heun's method - RK2 with virtual NFEs)
run_test "STORK-2" "mflux.contrib.schedulers.stork-2" ""

# Test 3: STORK-4 (RK4 with Taylor - highest quality)
run_test "STORK-4" "mflux.contrib.schedulers.stork-4" ""

# Test 4: ER-SDE Beta (deterministic beta distribution)
run_test "ER-SDE Beta" "mflux.contrib.schedulers.er_sde_beta" ""

# Test 5: ER-SDE Beta with SDE component (natural appearance)
run_test "ER-SDE Beta SDE" "mflux.contrib.schedulers.er_sde_beta" '{"gamma": 0.3, "beta_strength": 1.0}'

# Test 6: Advanced - Cosine (smooth transitions)
run_test "Advanced Cosine" "mflux.contrib.schedulers.advanced" '{"schedule": "cosine"}'

# Test 7: Advanced - Exponential (fast early, refined end)
run_test "Advanced Exponential" "mflux.contrib.schedulers.advanced" '{"schedule": "exponential"}'

# Test 8: Advanced - Sqrt (structure preservation)
run_test "Advanced Sqrt" "mflux.contrib.schedulers.advanced" '{"schedule": "sqrt"}'

# Test 9: Advanced - Scaled Linear (adaptive scaling)
run_test "Advanced Scaled Linear" "mflux.contrib.schedulers.advanced" '{"schedule": "scaled_linear"}'

# Test 10: Advanced - Beta RES_2M (ComfyUI popular preset)
run_test "Advanced Beta RES_2M" "mflux.contrib.schedulers.advanced" '{"schedule": "beta", "beta_alpha": 2.0, "beta_beta": 1.0}'

# Test 11: Advanced - Beta Default (symmetric distribution)
run_test "Advanced Beta Default" "mflux.contrib.schedulers.advanced" '{"schedule": "beta", "beta_alpha": 0.6, "beta_beta": 0.6}'

echo "========================================="
echo "All tests completed!"
echo "========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Generated images:"
ls -lh "$OUTPUT_DIR"/*.png
echo ""
echo "To compare results, open all images:"
echo "open $OUTPUT_DIR/*.png"
