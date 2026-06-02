"use client"

import { useState } from "react"
import type { FormEvent } from "react"
import { Shield } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

export type Role = "Citizen" | "Journalist" | "Police" | "Authority"

interface LoginPageProps {
  onLogin: (role: Role) => void
}

export function LoginPage({ onLogin }: LoginPageProps) {
  const [role, setRole] = useState<Role>("Citizen")
  const [identifier, setIdentifier] = useState("")

  const getPlaceholder = (selectedRole: Role) => {
    switch (selectedRole) {
      case "Citizen": return "Enter Email or Phone"
      case "Journalist": return "Enter Media ID"
      case "Police": return "Enter Department ID"
      case "Authority": return "Enter Government ID"
      default: return "Enter ID"
    }
  }

  const handleContinue = (e: FormEvent) => {
    e.preventDefault()
    if (identifier.trim()) {
      onLogin(role)
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-background p-4">
      <Card className="w-full max-w-md border-border/50 bg-card/50 backdrop-blur-sm shadow-xl">
        <CardHeader className="space-y-4 items-center text-center pb-8">
          <div className="p-3 bg-primary/10 rounded-2xl w-fit">
            <Shield className="w-10 h-10 text-primary" />
          </div>
          <div className="space-y-2">
            <CardTitle className="text-2xl font-bold tracking-tight">
              Welcome to BharatShield
            </CardTitle>
            <CardDescription className="text-base text-muted-foreground">
              Secure access for verified users
            </CardDescription>
          </div>
        </CardHeader>

        <CardContent>
          <form onSubmit={handleContinue} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="role">Select Role</Label>
              <Select value={role} onValueChange={(val) => setRole(val as Role)}>
                <SelectTrigger id="role">
                  <SelectValue placeholder="Select a role" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Citizen">Citizen</SelectItem>
                  <SelectItem value="Journalist">Journalist</SelectItem>
                  <SelectItem value="Police">Police</SelectItem>
                  <SelectItem value="Authority">Authority</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="identifier">Identification</Label>
              <Input
                id="identifier"
                placeholder={getPlaceholder(role)}
                value={identifier}
                onChange={(e) => setIdentifier(e.target.value)}
                required
              />
            </div>

            <Button type="submit" className="w-full text-base py-5 mt-4">
              Continue
            </Button>

            <p className="text-center text-sm text-muted-foreground pt-4">
              Role verification will be implemented in future version
            </p>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}