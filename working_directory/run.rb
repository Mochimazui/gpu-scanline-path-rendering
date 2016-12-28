
# -------- -------- -------- -------- -------- -------- -------- --------
require 'FileUtils'

# -------- -------- -------- -------- -------- -------- -------- --------
$benchmark_files = {	
	"boston"    => { filename: "boston",    width: 1024, height: 917  },
	"car"       => { filename: "car",       width: 1024, height: 682  },
	"contour"   => { filename: "contour",   width: 1024, height: 1024 },
	"embrace"   => { filename: "embrace",   width: 1024, height: 1096 },
	"hawaii"    => { filename: "hawaii",    width: 1024, height: 843  },
	"paper-1"   => { filename: "paper-1",   width: 1024, height: 1325 },
	"paper-2"   => { filename: "paper-2",   width: 1024, height: 1325 },
	"paris-30k" => { filename: "paris-30k", width: 1096, height: 1060 },
	"paris-50k" => { filename: "paris-50k", width: 657,  height: 635  },
	"paris-70k" => { filename: "paris-70k", width: 470,  height: 453  },	
	"tiger"     => { filename: "tiger",     width: 1024, height: 1055 },
	"reschart"  => { filename: "reschart",  width: 1024, height: 624  },	
	"chord"     => { filename: "chord",     width: 1024, height: 1024 }	
}

# -------- -------- -------- -------- -------- -------- -------- --------
$selected_files = []

# -------- -------- -------- -------- -------- -------- -------- --------
$color_modes = [
	:rgb,
	:srgb,
]

# -------- -------- -------- -------- -------- -------- -------- --------
$samples = [
	8,
	32,
]

# -------- -------- -------- -------- -------- -------- -------- --------
$output_scale = [
	1,
	2,
]

# -------- -------- -------- -------- -------- -------- -------- --------
$gen_cmd_file = false

# -------- -------- -------- -------- -------- -------- -------- --------
$output_commands = []
$benchmark_commands = []
$step_timing_commands = []

# -------- -------- -------- -------- -------- -------- -------- --------
def system_till_true(cmd) 
	puts '', '> ' + cmd
	STDOUT.flush
	count = 1
	while !system(cmd) 
		puts '', ">! retry command (#{++count}) : #{cmd}"
		STDOUT.flush
	end
end

# -------- -------- -------- -------- -------- -------- -------- --------
def update_file_info(benchmark_file_list)

	benchmark_file_list = $benchmark_files.keys if(benchmark_file_list[0] == 'all') 
	
	benchmark_file_list.each do |k|
		f = $benchmark_files[k]
		f[:filename] = "#{k}"
		f[:rvg] = f[:rvg] || "./input/rvg/#{k}.rvg"
		$selected_files.push(f);
	end

end

# -------- -------- -------- -------- -------- -------- -------- --------
def update_commands()
	
	timestamp = Time.now.strftime("%m%d-%H%M%S")

	$benchmark_output_file = "./result/timing-" + timestamp + ".csv"
	$step_timing_output_file = "./result/step-timing-" + timestamp + ".csv"

	$selected_files.each do |file| 
	
		file_name = file[:filename]
		rvg = file[:rvg]
		vg2_input_file = rvg
		
		width_1 = file[:width]
		height_1 = file[:height]

		width_2 = file[:width2] || (width_1.to_i * 2)
		height_2 = file[:height2] || (height_1.to_i * 2)
		
		flip = false
		
		#
		vg2_program = '../x64/Release/gpu-scanline.exe'
	
		#output
		o_shared = "#{vg2_program} --srgb --c-m-cs --save-output-file --fit-to-window --fix-output-size --attach-timing-to #{$benchmark_output_file} --input-name #{file_name} -i #{vg2_input_file}"
		
		o_8_1    = "#{o_shared} --samples 8              --output-width #{width_1} --output-height #{height_1} -o ./output/#{file_name}-#{width_1}-#{height_1}-8.png" 
		o_8_2    = "#{o_shared} --samples 8              --output-width #{width_2} --output-height #{height_2} -o ./output/#{file_name}-#{width_2}-#{height_2}-8.png" 
		
		o_ms8_1  = "#{o_shared} --samples 8  --ms-output --output-width #{width_1} --output-height #{height_1} -o ./output/#{file_name}-#{width_1}-#{height_1}-8-ms.png" 
		o_ms8_2  = "#{o_shared} --samples 8  --ms-output --output-width #{width_2} --output-height #{height_2} -o ./output/#{file_name}-#{width_2}-#{height_2}-8-ms.png" 
		
		o_32_1   = "#{o_shared} --samples 32             --output-width #{width_1} --output-height #{height_1} -o ./output/#{file_name}-#{width_1}-#{height_1}-32.png" 
		o_32_2   = "#{o_shared} --samples 32             --output-width #{width_2} --output-height #{height_2} -o ./output/#{file_name}-#{width_2}-#{height_2}-32.png" 
		
		o_ms32_1 = "#{o_shared} --samples 32 --ms-output --output-width #{width_1} --output-height #{height_1} -o ./output/#{file_name}-#{width_1}-#{height_1}-32-ms.png" 
		o_ms32_2 = "#{o_shared} --samples 32 --ms-output --output-width #{width_2} --output-height #{height_2} -o ./output/#{file_name}-#{width_2}-#{height_2}-32-ms.png" 
		
		$output_commands.push(o_8_1)
		$output_commands.push(o_8_2)
		$output_commands.push(o_ms8_1)
		$output_commands.push(o_ms8_2)
		$output_commands.push(o_32_1)
		$output_commands.push(o_32_2)
		$output_commands.push(o_ms32_1)
		$output_commands.push(o_ms32_2)

		#timing
		t_shared = "#{vg2_program} --srgb --c-m-cs --benchmark --fit-to-window --fix-output-size --attach-timing-to #{$benchmark_output_file} --input-name #{file_name} -i #{vg2_input_file}"
		
		t_8_1    = "#{t_shared} --samples 8              --output-width #{width_1} --output-height #{height_1}" 
		t_8_2    = "#{t_shared} --samples 8              --output-width #{width_2} --output-height #{height_2}" 
		
		t_ms8_1  = "#{t_shared} --samples 8  --ms-output --output-width #{width_1} --output-height #{height_1}" 
		t_ms8_2  = "#{t_shared} --samples 8  --ms-output --output-width #{width_2} --output-height #{height_2}" 
		
		t_32_1   = "#{t_shared} --samples 32             --output-width #{width_1} --output-height #{height_1}" 
		t_32_2   = "#{t_shared} --samples 32             --output-width #{width_2} --output-height #{height_2}" 
		
		t_ms32_1 = "#{t_shared} --samples 32 --ms-output --output-width #{width_1} --output-height #{height_1}" 
		t_ms32_2 = "#{t_shared} --samples 32 --ms-output --output-width #{width_2} --output-height #{height_2}" 
		
		$benchmark_commands.push(t_8_1)
		$benchmark_commands.push(t_8_2)
		$benchmark_commands.push(t_ms8_1)
		$benchmark_commands.push(t_ms8_2)
		$benchmark_commands.push(t_32_1)
		$benchmark_commands.push(t_32_2)
		$benchmark_commands.push(t_ms32_1)
		$benchmark_commands.push(t_ms32_2)

		# step timing commands
		st_shared = "#{vg2_program} --srgb --c-m-cs --step-timing --fit-to-window --fix-output-size --attach-timing-to #{$step_timing_output_file} --input-name #{file_name} -i #{vg2_input_file}"
		
		st_ms8_1  = "#{st_shared} --samples 8  --ms-output --output-width #{width_1} --output-height #{height_1}" 
		st_ms8_2  = "#{st_shared} --samples 8  --ms-output --output-width #{width_2} --output-height #{height_2}" 
		
		st_ms32_1 = "#{st_shared} --samples 32  --ms-output --output-width #{width_1} --output-height #{height_1}" 
		st_ms32_2 = "#{st_shared} --samples 32  --ms-output --output-width #{width_2} --output-height #{height_2}" 
		
		$step_timing_commands.push(st_ms32_1)
		
	end

end

# -------- -------- -------- -------- -------- -------- -------- --------
# -------- -------- -------- -------- -------- -------- -------- --------
# -------- -------- -------- -------- -------- -------- -------- --------
# -------- -------- -------- -------- -------- -------- -------- --------

def run_output() 
	cmdfile = open('00-output.cmd', 'w')
	$output_commands.each do |cmd| 
		if $gen_cmd_file
			cmdfile.write cmd + "\n"
		else
			system_till_true(cmd) 
		end
	end
	cmdfile.close()
end

def run_benchmark()
	FileUtils.touch $benchmark_output_file
	cmdfile = open('01-benchmark.cmd', 'w')
	$benchmark_commands.each  do |cmd| 
		if $gen_cmd_file
			cmdfile.write cmd + "\n"
		else
			system_till_true(cmd) 
		end
	end
	cmdfile.close()
end

def run_step_timing() 
	FileUtils.touch $step_timing_output_file
	cmdfile = open('02-step_timing.cmd', 'w')	
	$step_timing_commands.each do |cmd| 
		if $gen_cmd_file
			cmdfile.write cmd + "\n"
		else
			system_till_true(cmd) 
		end
	end
	cmdfile.close()
end

# -------- -------- -------- -------- -------- -------- -------- --------
# -------- -------- -------- -------- -------- -------- -------- --------

output_mode = false
benchmark_mode = false
step_timing_mode = false

benchmark_file_list = []
test_file_list = []

def example()
    puts ""
	puts "use:"
	puts ""
	puts "    run.rb -file=car,hawaii -output"
	puts "    run.rb -file=car,hawaii -benchmark"
	puts "    run.rb -file=car,hawaii -step-timing"
	puts ""
	puts "    run.rb -file=all -output -benchmark -step-timing"
	puts "    run.rb -file=all -output -benchmark -step-timing -gen-cmd-file"
	puts ""
	puts "find output in ./output or ./result"
	exit
end

# -------- -------- -------- -------- -------- -------- -------- --------
if ARGV.size == 0
	example()
else
	ARGV.each do |a|
		l = a.split('=')
		l[1] = l[1] ? l[1].split(",") : nil
		
		case l[0] 
		when '-output'
			output_mode = true
		when '-benchmark'
			benchmark_mode = true
		when '-step-timing'
			step_timing_mode = true
		when '-file'
			benchmark_file_list = l[1] if l[1]
		when '-gen-cmd-file'
			$gen_cmd_file = true
		else
			example()
		end
	end
end

# -------- -------- -------- -------- -------- -------- -------- --------
if benchmark_file_list.empty?
	benchmark_file_list = ["all"]
end

update_file_info(benchmark_file_list)

# -------- -------- -------- -------- -------- -------- -------- --------
update_commands()

# -------- -------- -------- -------- -------- -------- -------- --------
start_time = Time.now

run_output() if output_mode
run_benchmark() if benchmark_mode
run_step_timing if step_timing_mode

end_time = Time.now
puts "Total time: #{end_time - start_time}s"

puts ""
puts "find output in ./output or ./result"
