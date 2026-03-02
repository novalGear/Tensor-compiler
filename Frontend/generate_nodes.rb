#!/usr/bin/env ruby
# frozen_string_literal: true

require_relative 'generator_lib'
require 'fileutils'

SCRIPT_DIR = File.dirname(__FILE__)
JSON_FILE = File.join(SCRIPT_DIR, 'ops_def.json')

OUTPUT_HPP = File.join(SCRIPT_DIR, 'Graph', 'graph_gen.hpp')
OUTPUT_INL = File.join(SCRIPT_DIR, 'Graph', 'graph_gen_parser.inl')
OUTPUT_UTILS = File.join(SCRIPT_DIR, 'Graph', 'graph_gen_utils.inl')

# Чтение данных
unless File.exist?(JSON_FILE)
  puts "Error: #{JSON_FILE} not found!"
  exit 1
end

puts "Reading #{JSON_FILE}..."
data = JSON.parse(File.read(JSON_FILE))

common_def = data.find { |op| op['op_type'] == 'CommonFields' }
common_fields = common_def ? common_def['fields'] : []
real_ops = data.reject { |op| op['op_type'] == 'CommonFields' }

puts "Found CommonFields: #{common_fields.map { |f| f['name'] }.join(', ')}"
puts "Generating code for #{real_ops.size} operators..."

# 1. Генерация graph_gen.hpp
puts "Writing #{OUTPUT_HPP}..."
File.open(OUTPUT_HPP, 'w') do |f|
  NodeGenerator.write_header_start(f)
  real_ops.each { |op| NodeGenerator.write_node_struct(f, op, common_fields) }
  NodeGenerator.write_variant_typedef(f, data)
  NodeGenerator.write_header_end(f)
end

# 2. Генерация graph_gen_parser.inl
puts "Writing #{OUTPUT_INL}..."
File.open(OUTPUT_INL, 'w') do |f|
  NodeGenerator.write_parser_start(f)
  real_ops.each do |op|
    NodeGenerator.write_parse_attributes_function(f, op)
    NodeGenerator.write_factory_function(f, op, common_fields)
  end
  NodeGenerator.write_dispatcher_function(f, data)
  NodeGenerator.write_parser_end(f)
end

# 3. Генерация graph_gen_utils.inl (для DOT дампа)
puts "Writing #{OUTPUT_UTILS}..."
File.open(OUTPUT_UTILS, 'w') do |f|
  NodeGenerator.write_dot_helper_function(f, data, common_fields)
end

puts "Done! Generated:"
puts "  - #{OUTPUT_HPP}"
puts "  - #{OUTPUT_INL}"
puts "  - #{OUTPUT_UTILS}"
