"use strict";
import { v3norm } from '../math/vec3.js';

export const classDefs = [
  { label:"A", color:"#534AB7", dir:v3norm([0.75,0.55,0.35]), kappa:80, n:200, aniso:false },
  { label:"B", color:"#1D9E75", dir:v3norm([-0.35,0.8,0.5]), kappa:30, n:80, aniso:false },
  { label:"C", color:"#D85A30", dir:v3norm([0.5,-0.65,0.55]), kappa:80, n:15, aniso:false },
  { label:"D", color:"#D4537E", dir:v3norm([-0.55,-0.35,0.75]), kappa:30, n:80, aniso:true, stretch:3.0 },
  { label:"E", color:"#378ADD", dir:v3norm([0.1,0.6,0.8]), kappa:30, n:60, aniso:false },
];

export const classNotes = {
  A: "",
  B: "",
  C: "few-shot",
  D: "stretched \u00d73 along one tangent axis",
  E: "direction near B"
};
